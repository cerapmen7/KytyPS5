#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfProbe.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <xxhash.h>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderDepthInfo& depth,
                               const CommandBuffer& buffer, uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF("DrawFramebufferSkip[%u]: %s depth_format=%s depth_image=%s depth_vaddr_num=%d "
	     "target_mask=0x%08" PRIx32 " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	     log_id, draw_name, vk::to_string(depth.desc.view_info.format).c_str(),
	     depth.image_id ? "yes" : "no",
	     static_cast<int>(!depth.desc.info.data.Empty()) +
	         static_cast<int>(depth.desc.info.HasStencil()),
	     ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.module->info.info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct DrawCallInfo {
	bool     indexed        = true;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	uint64_t index_addr     = 0;

	[[nodiscard]] const char* Name() const { return indexed ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t    address            = 0;
	const void* host_data          = nullptr;
	uint64_t    size               = 0;
	uint32_t    guest_element_size = 0;
};

// The task shader's inputs for an indirect mesh draw: Params in gpu_mesh_indirect.task.
struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	vk::DeviceAddress index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	// At most RES_MAX (32) ranges: an insertion sort beats std::sort's partitioning here and
	// this runs once per draw.
	for (uint32_t i = 1; i < range_count; i++) {
		const auto range = ranges[i];
		uint32_t   j     = i;
		while (j != 0 && ranges[j - 1].base_address > range.base_address) {
			ranges[j] = ranges[j - 1];
			j--;
		}
		ranges[j] = range;
	}

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// Opt-in performance experiment: KYTY_SKIP_DEPTH_ONLY=1 drops the shadow/depth-only passes.
// Read once; off unless the lab asks for it.
static bool SkipDepthOnlyEnabled() {
	static const bool enabled = [] {
		const auto* const value = std::getenv("KYTY_SKIP_DEPTH_ONLY");
		return value != nullptr && *value != '\0' && std::strtoul(value, nullptr, 10) != 0;
	}();
	return enabled;
}

// Opt-in companion: KYTY_SKIP_DEPTH_EARLY=1 decides the depth-only skip before the per-draw
// metadata resolution and the executor lock. The later check stays in place, so a draw that
// survives this one is still skipped where it was before.
static bool EarlyDepthSkipEnabled() {
	static const bool enabled = [] {
		const auto* const value = std::getenv("KYTY_SKIP_DEPTH_EARLY");
		return value != nullptr && *value != '\0' && std::strtoul(value, nullptr, 10) != 0;
	}();
	return enabled;
}

// Mirrors the colour-output mask PipelineCache::GetGraphicsPrograms derives from the shader
// registers, so a pass that exports no colour is recognised before any program work.
static uint32_t RegisterColorOutputMask(const HW::Context& context) {
	const auto& sh         = context.GetShaderRegisters();
	const auto  write_mask = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    mask       = context.HasObjectPrimitiveIdOutput() ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			mask |= 1u << slot;
		}
	}
	return mask;
}

// A draw that exports no colour and writes depth only feeds shadow cascades and depth
// prepasses; it cannot change the presented image by itself.
static bool IsDepthOnlyDraw(const CommandBuffer& buffer) {
	const auto& context = buffer.GetRegisters();
	if (RegisterColorOutputMask(context) != 0) {
		return false;
	}
	const auto& z = context.GetDepthRenderTarget();
	return z.z_write_base_addr != 0 || z.stencil_write_base_addr != 0;
}

bool RenderExecutor::ShouldSkipDepthOnly(CommandBuffer& buffer) {
	const int  mode        = DepthCacheMode();
	const bool static_mode = mode == 3 || mode == 6;
	const bool skip        = SkipDepthOnlyEnabled() || (static_mode && m_static_skip);
	if (!skip || !IsDepthOnlyDraw(buffer)) {
		return false;
	}
	const auto& z = buffer.GetRegisters().GetDepthRenderTarget();
	if (mode == 3 || mode == 6) {
		if (!m_static_skip) {
			return false;
		}
		if (mode == 6) {
			// The screen-sized depth prepare pass feeds the colour pass depth test. Replaying it
			// from an older frame would reject pixels of animated geometry, so it always renders.
			const uint32_t width  = static_cast<uint32_t>(z.size.x_max) + 1u;
			const uint32_t height = static_cast<uint32_t>(z.size.y_max) + 1u;
			if (width == Config::GetScreenWidth() && height == Config::GetScreenHeight() &&
			    z.depth_view.slice_max == z.depth_view.slice_start) {
				return false;
			}
		}
		ResolveRenderDepthTarget(buffer, m_targets.depth);
		const auto target_id = m_targets.depth.image_id;
		if (!target_id) {
			return false;
		}
		auto& target = m_context.GetTextureCache().GetImage(target_id);
		if (!target.registered) {
			return false;
		}
		const auto key = StaticSnapshotKey(target.info);
		const auto snapshot =
		    std::ranges::find(m_static_snapshots, key, &StaticSnapshot::key);
		if (snapshot == m_static_snapshots.end() || snapshot->image == nullptr) {
			return false;
		}
		// Restore the exact pixels last written by a full render before the colour pass samples
		// this target. Re-copy only when the target changed since the previous restore.
		const auto restored = m_restored_versions.find(target_id.index);
		if (restored == m_restored_versions.end() || restored->second != target.Version()) {
			target.CopyImage(*snapshot->image);
			m_restored_versions[target_id.index] = target.Version();
		}
		m_static_frame_skipped = true;
	}
	if (mode == 4) {
		// Mode 4 keeps the screen-sized depth prepare pass; only off-screen/arrayed shadow
		// cascades are replayed. Skipping the main depth target breaks the colour pass.
		const uint32_t width  = static_cast<uint32_t>(z.size.x_max) + 1u;
		const uint32_t height = static_cast<uint32_t>(z.size.y_max) + 1u;
		if (width == Config::GetScreenWidth() && height == Config::GetScreenHeight() &&
		    z.depth_view.slice_max == z.depth_view.slice_start) {
			return false;
		}
	}
	if (mode == 5) {
		if (m_sampled_ranges_prev.empty()) {
			return false;
		}
		const uint32_t width  = static_cast<uint32_t>(z.size.x_max) + 1u;
		const uint32_t height = static_cast<uint32_t>(z.size.y_max) + 1u;
		if (width == Config::GetScreenWidth() && height == Config::GetScreenHeight() &&
		    z.depth_view.slice_max == z.depth_view.slice_start) {
			return false;
		}
		// Only skip cascades the colour pass did not sample last frame. They cannot affect the
		// presented image, so this is exact rather than a temporal approximation.
		for (const auto& [address, size]: m_sampled_ranges_prev) {
			if (size != 0 && z.z_write_base_addr >= address &&
			    z.z_write_base_addr - address < size) {
				return false;
			}
		}
	}
	if (mode >= 3) {
		// Only a draw that is actually skipped may consume the guest fast-clear bookkeeping;
		// consuming it for a rendered shadow pass removes its clear and lights the scene wrong.
		auto& cache = m_context.GetTextureCache();
		if (z.htile_data_base_addr != 0 &&
		    cache.IsMetaCleared(z.htile_data_base_addr, z.depth_view.slice_start)) {
			(void)cache.TouchMeta(z.htile_data_base_addr, z.depth_view.slice_start, false);
		}
	}
	if (mode == 3 || mode == 6) {
		// ResolveRenderDepthTarget marked the target as an attachment. The draw is skipped, so
		// clear that state before the colour pass samples it as a texture.
		ResetBindings();
	}
	return true;
}

// Opt-in shadow/depth replay cache. Unlike KYTY_SKIP_DEPTH_ONLY it never removes a pass: the
// first frame renders every depth draw normally, and a later frame reuses that frame's target
// pixels only while the draw's registers and every guest buffer/texture it reads are unchanged.
// A changed input falls back to a full render, so the visible scene is identical.
bool RenderExecutor::DepthCacheEnabled() {
	return DepthCacheMode() != 0;
}

int RenderExecutor::DepthCacheMode() {
	static const int mode = [] {
		const auto* const value = std::getenv("KYTY_DEPTH_CACHE");
		return (value != nullptr && *value != '\0') ? static_cast<int>(std::strtol(value, nullptr, 10))
		                                            : 0;
	}();
	return mode;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

uint64_t RenderExecutor::StaticSnapshotKey(const ImageInfo& info) {
	uint64_t key = XXH3_64bits(&info.data.address, sizeof(info.data.address));
	key          = XXH3_64bits_withSeed(&info.data.size, sizeof(info.data.size), key);
	key          = XXH3_64bits_withSeed(&info.pixel_format, sizeof(info.pixel_format), key);
	key          = XXH3_64bits_withSeed(&info.extent, sizeof(info.extent), key);
	key          = XXH3_64bits_withSeed(&info.resources, sizeof(info.resources), key);
	key          = XXH3_64bits_withSeed(&info.samples, sizeof(info.samples), key);
	return key;
}

static bool StaticGameInputDown() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Keys held in another application must not keep the game's replay disabled.
	const auto foreground = GetForegroundWindow();
	DWORD      owner      = 0;
	GetWindowThreadProcessId(foreground, &owner);
	if (owner != GetCurrentProcessId()) {
		return false;
	}
	// The launcher maps the pad to these keys; any held key means the scene is being driven.
	static const int keys[] = {
	    'X', 'L', 'K', 'I', 'Z', 'S', 'Q', 'D', 'T', 'G', 'F', 'H',
	    VK_RETURN, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_PRIOR, VK_NEXT, VK_HOME, VK_END, VK_SPACE,
	};
	for (const int key: keys) {
		if ((GetAsyncKeyState(key) & 0x8000) != 0) {
			return true;
		}
	}
#endif
	return false;
}

static bool DepthCacheLogEnabled() {
	static const bool enabled = [] {
		const auto* const value = std::getenv("KYTY_DEPTH_CACHE_LOG");
		return value != nullptr && *value != '\0' && std::strtoul(value, nullptr, 10) != 0;
	}();
	return enabled;
}

uint64_t RenderExecutor::DepthStateHash(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                        DepthCacheDraw* parts) {
	const auto hash_bytes = [](std::span<const uint8_t> bytes, uint64_t seed) {
		return XXH3_64bits_withSeed(bytes.data(), bytes.size(), seed);
	};
	const auto& regs = buffer.GetRegisters();
	parts->state_ctx = hash_bytes(regs.DepthStateBytes(), 0x9e3779b97f4a7c15ull);
	parts->state_cfg = hash_bytes(buffer.GetUserConfig().RawBytes(), 0x517cc1b727220a95ull);
	// The Shader object also carries compute and pixel stage registers. Compute state changes
	// between graphics draws and pixel state is stale when the depth draw has no pixel shader;
	// hashing them would reject every replay. Keep the vertex stage always and the pixel stage
	// only when it can affect this draw's output.
	const auto& shaders = buffer.GetShaders();
	const auto& sh_regs = regs.GetShaderRegisters();
	const auto& db      = sh_regs.db_shader_control;
	const bool  ps_active =
	    shaders.GetPs().ps_regs.data_addr != 0 &&
	    (RegisterColorOutputMask(regs) != 0 || db.shader_kill_enable ||
	     db.shader_z_export_enable || db.shader_mask_export_enable ||
	     db.shader_dual_export_enable || db.shader_execute_on_noop || sh_regs.AlphaToMask());
	// User SGPR arrays carry per-draw resource-table bases and inline constants. They move
	// between frames even when the emitted geometry is identical, so the register shape and
	// shader code identify the stage; the bound buffer contents are compared separately.
	HW::VertexShaderInfo vs = shaders.GetVs();
	vs.hs_user_sgpr         = {};
	vs.gs_user_sgpr         = {};
	vs.hs_regs.user_data_addr = 0;
	vs.gs_regs.user_data_addr = 0;
	parts->state_shader =
	    XXH3_64bits_withSeed(&vs, sizeof(vs), 0x6eed0e9da4d94a4full);
	if (ps_active) {
		HW::PixelShaderInfo ps = shaders.GetPs();
		ps.ps_user_sgpr        = {};
		parts->state_shader = XXH3_64bits_withSeed(&ps, sizeof(ps), parts->state_shader);
	}
	const uint64_t line[2] = {std::bit_cast<uint32_t>(regs.GetLineWidth()),
	                          regs.GetPrimitiveResetIndex()};
	parts->state_line      = XXH3_64bits(line, sizeof(line));
	const uint64_t args[4] = {draw.indexed ? 1u : 0u, draw.index_count, draw.instance_count,
	                          draw.index_addr};
	parts->state_args      = XXH3_64bits(args, sizeof(args));
	parts->state_hash      = XXH3_64bits_withSeed(&parts->state_ctx, sizeof(uint64_t) * 5,
	                                              0x27d4eb2f165667c5ull);
	return parts->state_hash;
}

uint64_t RenderExecutor::DepthTargetHash(const HW::DepthRenderTarget& target, bool clear) {
	uint64_t seed = clear ? 0x9e3779b97f4a7c15ull : 0x517cc1b727220a95ull;
	return XXH3_64bits_withSeed(&target, sizeof(target), seed);
}

uint64_t RenderExecutor::DepthRangeHash(uint64_t address, uint64_t size,
                                        const DepthCacheDraw* previous, uint32_t index) {
	constexpr uint64_t kMaxRange = 64ull * 1024 * 1024;
	if (address == 0 || size == 0 || size > kMaxRange) {
		return 0;
	}
	auto& buffer_cache = m_context.GetBufferCache();
	// Reading a range with pending GPU writes would make the tracker download it to the CPU
	// and stall the frame. Merely being GPU-owned is fine once the bytes are current; such a
	// draw can still be hashed without a readback.
	if (buffer_cache.HasGpuDirtyBytes(address, size)) {
		return 0;
	}
	const bool cpu_modified = buffer_cache.IsRegionCpuModified(address, size);
	// Static ranges repeat across hundreds of draws (shared vertex data); hash them once per
	// frame instead of once per draw. Dirty ranges are small and always rehashed, so a guest
	// rewrite is still observed.
	const uint64_t key = address ^ (size * 0x9e3779b97f4a7c15ull);
	if (!cpu_modified && !m_depth_range_dirty.contains(key)) {
		if (previous != nullptr && index < previous->ranges.size()) {
			const auto& old = previous->ranges[index];
			if (old.address == address && old.size == size) {
				return old.hash;
			}
		}
		const auto it = m_depth_range_cache.find(key);
		if (it != m_depth_range_cache.end() && it->second.address == address &&
		    it->second.size == size) {
			return it->second.hash;
		}
		const auto hash = XXH3_64bits(reinterpret_cast<const void*>(address), size);
		if (m_depth_range_cache.size() < 16384) {
			m_depth_range_cache.emplace(key, DepthCacheRange {address, size, hash});
		}
		return hash;
	}
	// A range that was dirty once this frame may have been uploaded and the dirty bit cleared;
	// do not trust an older hash for the rest of the frame.
	m_depth_range_dirty.insert(key);
	return XXH3_64bits(reinterpret_cast<const void*>(address), size);
}

struct DepthTimer {
	uint64_t& target;
	uint64_t  start = Common::PerfProbe::Now();
	explicit DepthTimer(uint64_t& ns): target(ns) {}
	~DepthTimer() { target += Common::PerfProbe::Now() - start; }
};

bool RenderExecutor::TryReplayDepthDraw(CommandBuffer& buffer, const DrawCallInfo& draw) {
	m_depth_record = false;
	if (!DepthCacheEnabled()) {
		return false;
	}
	if (DepthCacheMode() >= 3) {
		// Static modes only gate the existing depth-only skip; the decision is made in DrawIndex.
		return false;
	}
	if (DepthCacheMode() >= 2) {
		// Mode 2 compares the PM4 packets that set up this draw. The hash covers every register
		// write and every DMA copy since the previous draw, so an identical hash means the
		// draw is fed by exactly the same command stream. The colour pass itself always runs.
		if (!IsDepthOnlyDraw(buffer)) {
			return false;
		}
		const auto& hw = buffer.GetRegisters();
		const auto& rc = hw.GetRenderControl();
		if (rc.copy_depth_to_color || rc.copy_stencil_to_color) {
			return false;
		}
		const uint64_t hash      = buffer.GetCommandHash();
		const auto     ordinal   = static_cast<uint32_t>(m_depth_hash_current.size());
		m_depth_hash_current.push_back(hash);
		if (!m_depth_hash_skip_disabled && ordinal < m_depth_hash_prev.size() &&
		    m_depth_hash_prev[ordinal] == hash && hash != 0) {
			m_depth_hash_skip_count++;
			return true;
		}
		m_depth_hash_render_count++;
		// A cleared target or a changed stream invalidates every later draw in this frame:
		// rendering one draw after a clear would otherwise leave the skipped pixels missing.
		m_depth_hash_skip_disabled = true;
		return false;
	}
	if (!IsDepthOnlyDraw(buffer)) {
		// The first draw of the next pass closes the cached one. Pinning its target version here
		// catches a later color pass that writes the same depth image: its pixels are not the
		// pass output the cache would reuse.
		if (!m_depth_frame.empty() && m_depth_frame.back().target_version == 0 &&
		    m_depth_frame.back().target_id) {
			auto& pass          = m_depth_frame.back();
			pass.target_version = m_context.GetTextureCache().GetImage(pass.target_id).Version();
		}
		return false;
	}
	DepthTimer timer(m_depth_try_ns);
	const auto& hw = buffer.GetRegisters();
	const auto& rc = hw.GetRenderControl();
	if (rc.copy_depth_to_color || rc.copy_stencil_to_color) {
		return false;
	}
	if (m_context.GetOcclusionQueries().Open()) {
		return false;
	}
	// Use the raw depth-target registers as the pass key. Resolving the host image here would
	// bind the target twice per draw and change its layout tracking; the prepared draw records
	// the host ImageId on the first execution instead.
	const auto& z = hw.GetDepthRenderTarget();
	if (z.z_info.format == Prospero::DepthFormat::kInvalid || z.z_write_base_addr == 0 ||
	    z.z_info.max_mip_level != 0 || z.z_info.partially_resident) {
		m_depth_uncacheable = true;
		m_depth_skip_rest   = true;
		return false;
	}
	const bool htile = z.z_info.htile_acceleration || z.htile_data_base_addr != 0;
	if (z.stencil_info.format != Prospero::StencilFormat::kInvalid) {
		m_depth_reason_stencil++;
	}
	const bool meta_clear =
	    htile && m_context.GetTextureCache().IsMetaCleared(z.htile_data_base_addr,
	                                                       z.depth_view.slice_start);
	const uint64_t target_hash = DepthTargetHash(z, rc.depth_clear_enable);
	const bool     clear       = rc.depth_clear_enable;
	bool           new_pass    = m_depth_frame.empty();
	if (!new_pass) {
		auto& last = m_depth_frame.back();
		if (last.target_hash != target_hash) {
			new_pass = true;
		} else if (clear && !last.draws.empty()) {
			// A second explicit clear of the same target starts a second pass.
			new_pass = true;
		}
	}
	if (new_pass) {
		if (!m_depth_frame.empty()) {
			auto& previous = m_depth_frame.back();
			if (previous.target_id && previous.target_version == 0) {
				previous.target_version =
				    m_context.GetTextureCache().GetImage(previous.target_id).Version();
			}
		}
		DepthCachePass pass;
		pass.target_hash = target_hash;
		const uint32_t pass_index = static_cast<uint32_t>(m_depth_frame.size());
		pass.cacheable            = true;
		if (m_depth_cache_ready) {
			if (pass_index >= m_depth_cache.size()) {
				pass.skip_rest = true;
				m_depth_reason_pass++;
			} else {
				const auto& cached = m_depth_cache[pass_index];
				const auto& image  = m_context.GetTextureCache().GetImage(cached.target_id);
				if (!cached.cacheable) {
					// Rebuild a pass that was a placeholder last frame.
					pass.skip_rest = true;
					m_depth_reason_cached_invalid++;
				} else if (!cached.target_id || !image.registered ||
				           cached.target_hash != target_hash) {
					pass.skip_rest = true;
					m_depth_reason_target++;
				} else {
					pass.target_id = cached.target_id;
				}
			}
		}
		m_depth_frame.push_back(std::move(pass));
	}
	const uint32_t pass_index = static_cast<uint32_t>(m_depth_frame.size() - 1);
	auto&          pass       = m_depth_frame[pass_index];
	const bool     first_draw = pass.draws.empty();
	DepthCacheDraw state_parts;
	const uint64_t state_hash = DepthStateHash(buffer, draw, &state_parts);
	const bool can_compare = m_depth_cache_ready && pass.cacheable && !pass.skip_rest &&
	                         pass_index < m_depth_cache.size();
	if (can_compare) {
		const auto& cached = m_depth_cache[pass_index];
		const auto  index  = static_cast<uint32_t>(pass.draws.size());
		m_depth_reason_compared++;
		if (index < cached.draws.size()) {
			const auto& old          = cached.draws[index];
			bool        state_match  = old.cacheable && old.state_hash == state_hash;
			bool        ranges_match = true;
			bool        images_match = true;
			if (old.cacheable && !state_match) {
				m_depth_reason_state++;
				if (old.state_ctx != state_parts.state_ctx) m_depth_reason_state_ctx++;
				if (old.state_cfg != state_parts.state_cfg) m_depth_reason_state_cfg++;
				if (old.state_shader != state_parts.state_shader) m_depth_reason_state_shader++;
				if (old.state_args != state_parts.state_args) m_depth_reason_state_args++;
				if (old.state_line != state_parts.state_line) m_depth_reason_state_line++;
			}
			for (uint32_t i = 0; state_match && i < old.range_count; i++) {
				if (DepthRangeHash(old.ranges[i].address, old.ranges[i].size, &old, i) !=
				    old.ranges[i].hash) {
					ranges_match = false;
					break;
				}
			}
			for (uint32_t i = 0; images_match && i < old.image_count; i++) {
				const auto& image = m_context.GetTextureCache().GetImage(old.images[i].id);
				// The version also moves when a sampled image refreshes cache bookkeeping. Only
				// an actual CPU or GPU content change must stop the replay.
				images_match = old.images[i].id && image.registered && !image.IsCpuDirty() &&
				               !image.IsGpuModified();
			}
			if (!ranges_match) m_depth_reason_range++;
			if (!images_match) m_depth_reason_target++;
			bool match = state_match && ranges_match && images_match;
			if (match) {
				if (meta_clear &&
				    !m_context.GetTextureCache().TouchMeta(z.htile_data_base_addr,
				                                           z.depth_view.slice_start, false)) {
					match = false;
					m_depth_reason_meta++;
				} else {
					pass.draws.push_back(old);
					pass.skipped = true;
					m_depth_skip_count++;
					return true;
				}
			}
		} else {
			m_depth_reason_pass++;
		}
	}
	// If the pass's first draw executes, its clear wipes the pixels the later draws would have
	// replayed; render the whole pass instead. Skipping the first draw suppresses its clear,
	// which keeps the cached contents valid.
	if (first_draw && clear) {
		pass.skip_rest = true;
	}
	// The draw executes. Record its inputs so the next frame can compare them. A pass that is
	// not promotable still records placeholders, keeping every later draw index aligned.
	m_depth_pending      = state_parts;
	m_depth_pending_hash = state_hash;
	m_depth_pending_pass = pass_index;
	m_depth_record       = true;
	return false;
}

void RenderExecutor::ReplayDepthRecordDraw(const DrawRenderState& state, const DrawEmitInfo& emit,
                                           const DrawIndexBufferSource& index_source,
                                           std::span<PreparedBindings* const> stages) {
	if (!m_depth_record || m_depth_pending_pass >= m_depth_frame.size()) {
		m_depth_record = false;
		return;
	}
	DepthTimer timer(m_depth_record_ns);
	m_depth_record = false;
	auto& pass      = m_depth_frame[m_depth_pending_pass];
	auto& draw      = m_depth_pending;
	draw.state_hash = m_depth_pending_hash;
	bool ok             = true;
	bool reason_writes  = false;
	bool reason_range   = false;
	if (m_targets.depth.image_id) {
		pass.target_id = m_targets.depth.image_id;
	} else {
		ok = false;
	}

	const DepthCacheDraw* previous = nullptr;
	if (m_depth_pending_pass < m_depth_cache.size()) {
		const auto& old_pass = m_depth_cache[m_depth_pending_pass];
		if (pass.draws.size() < old_pass.draws.size()) {
			previous = &old_pass.draws[pass.draws.size()];
		}
	}
	auto add_range = [&](uint64_t address, uint64_t size) {
		if (address == 0 || size == 0) {
			return;
		}
		if (size > 64ull * 1024 * 1024) {
			ok           = false;
			reason_range = true;
			return;
		}
		if (draw.range_count >= DepthCacheDraw::MaxRanges) {
			ok           = false;
			reason_range = true;
			return;
		}
		const auto index = draw.range_count;
		auto       hash  = DepthRangeHash(address, size, previous, index);
		if (hash == 0) {
			ok           = false;
			reason_range = true;
			return;
		}
		draw.ranges[draw.range_count++] = {address, size, hash};
	};
	const auto& vs = state.vertex_info[0];
	for (uint32_t i = 0; i < static_cast<uint32_t>(vs.buffers_num); i++) {
		const auto& b    = vs.buffers[i];
		const auto  size = static_cast<uint64_t>(b.stride) * b.num_records;
		add_range(b.addr, size);
	}

	if (index_source.address != 0 && index_source.size != 0) {
		add_range(index_source.address, index_source.size);
	}
	if (emit.indirect.address != 0 && emit.indirect.count != 0) {
		add_range(emit.indirect.address,
		          static_cast<uint64_t>(emit.indirect.count) * emit.indirect.stride);
	}
	if (vs.stage.resources != nullptr) {
		for (const auto& read: vs.stage.resources->specialization_reads) {
			add_range(read.first, read.second);
		}
	}
	for (auto* stage: stages) {
		for (const auto& source: stage->buffer_sources) {
			add_range(source.address, source.size);
		}
		if (HasShaderBufferWrites(*stage->runtime)) {
			ok            = false;
			reason_writes = true;
		}
		for (const auto& binding: stage->images) {
			if (!binding.image_id) continue;
			if (binding.image_id == m_targets.depth.image_id) {
				ok            = false;
				reason_writes = true;
				continue;
			}
			if (draw.image_count >= DepthCacheDraw::MaxImages) {
				ok = false;
				continue;
			}
			draw.images[draw.image_count++] = {
			    binding.image_id, m_context.GetTextureCache().GetImage(binding.image_id).Version()};
		}
	}
	if (!ok) {
		// This draw's inputs cannot be compared without a readback. Render it and the rest of
		// its pass, and do not promote that pass; other passes stay untouched.
		DepthCacheDraw sentinel;
		sentinel.cacheable = false;
		pass.draws.push_back(sentinel);
		m_depth_frame_draws++;
		// Depth-only writes are independent: keep the pass replayable and let the neighbouring
		// draws skip. The sentinel itself renders every frame.
		m_depth_reason_uncacheable++;
		if (reason_writes) m_depth_reason_writes++;
		if (reason_range) m_depth_reason_range++;
		return;
	}
	draw.cacheable = true;
	pass.draws.push_back(std::move(draw));
	m_depth_frame_draws++;
}

void RenderExecutor::DepthCacheEndFrame() {
	if (!DepthCacheEnabled()) {
		return;
	}
	DepthTimer end_timer(m_depth_end_ns);
	// A frame can flip more than once (layers); only the first boundary after draws counts.
	if (DepthCacheMode() >= 3) {
		if (m_static_frame_draws == 0) {
			return;
		}
	} else if (m_depth_frame.empty() && m_depth_frame_draws == 0 &&
	           m_depth_hash_current.empty()) {
		return;
	}
	if (DepthCacheMode() >= 3) {
		const auto shaders = Common::DebugCounters::Get(
		    Common::DebugCounters::Counter::ShadersCompiled);
		const bool shader_stable = shaders == m_static_last_shaders;
		m_static_last_shaders     = shaders;
		m_static_input            = StaticGameInputDown();
		if (m_static_input) {
			m_static_skip    = false;
			m_static_idle    = 0;
			m_static_refresh = 0;
		} else if (m_static_frame_draws >= 1000 && shader_stable) {
			if (m_static_idle < 100000) {
				m_static_idle++;
			}
			if (m_static_idle >= 150) {
				m_static_skip = true;
			}
		} else {
			m_static_skip    = false;
			m_static_idle    = 0;
			m_static_refresh = 0;
		}
		// Refresh the cached shadow maps periodically so idle animations do not freeze.
		if (m_static_skip) {
			if (++m_static_refresh >= 48) {
				m_static_skip    = false;
				m_static_refresh = 0;
			}
		}
		if ((DepthCacheMode() == 3 || DepthCacheMode() == 6) && !m_static_frame_skipped &&
		    (m_static_idle >= 120 || m_static_skip)) {
			// New targets (including the second buffer of a shadow-map ping-pong) are captured
			// during the last idle frames before activation. Nothing is copied while the level
			// is loading. Existing snapshots refresh only on activation/periodic-refresh frames.
			const bool refresh_snapshots = m_static_skip;
			const auto update_snapshot   = [&](ImageId id) {
				auto& target = m_context.GetTextureCache().GetImage(id);
				if (!target.registered) {
					return;
				}
				const auto key = StaticSnapshotKey(target.info);
				auto       snapshot =
				    std::ranges::find(m_static_snapshots, key, &StaticSnapshot::key);
				if (snapshot == m_static_snapshots.end()) {
					ImageInfo info = target.info;
					info.data      = {};
					info.stencil   = {};
					info.metadata  = {};
					auto image     = std::make_shared<Image>(m_context.GetGraphics(),
					                                         m_context.GetCommandScheduler(), info);
					m_static_snapshots.push_back({key, std::move(image)});
					snapshot = std::prev(m_static_snapshots.end());
				} else if (!refresh_snapshots) {
					return;
				}
				if (snapshot->image != nullptr) {
					snapshot->image->CopyImage(target);
				}
			};
			for (const auto id: m_depth_target_ids) {
				update_snapshot(id);
			}
			for (const auto id: m_sampled_ids) {
				update_snapshot(id);
			}
		}
		if (!m_static_frame_skipped) {
			m_sampled_ids_prev = m_sampled_ids;
		}
		if (DepthCacheMode() == 5) {
			m_sampled_ranges_prev.clear();
			for (const auto id: m_sampled_ids) {
				const auto& image = m_context.GetTextureCache().GetImage(id);
				if (!image.registered || !image.info.IsDepth() || image.info.data.size == 0) {
					continue;
				}
				const auto address = image.info.data.address;
				bool       duplicate = false;
				for (const auto& [old_address, old_size]: m_sampled_ranges_prev) {
					(void)old_size;
					if (old_address == address) {
						duplicate = true;
						break;
					}
				}
				if (!duplicate) {
					m_sampled_ranges_prev.emplace_back(address, image.info.data.size);
				}
			}
		}
		m_sampled_ids.clear();
		m_sampled_seen.clear();
		m_depth_target_ids.clear();
		m_sampled_restored = false;
		if (!m_static_skip || !m_static_frame_skipped) {
			m_restored_versions.clear();
		}
		m_static_frame_skipped = false;
		if (DepthCacheLogEnabled()) {
			static uint32_t log_frame = 0;
			if ((++log_frame % 60u) == 0) {
				std::printf("STATIC_CACHE mode=%d idle=%u skip=%d input=%d draws=%llu "
				            "shaders=%llu sampled=%zu snapshots=%zu\n",
				            DepthCacheMode(), m_static_idle, m_static_skip ? 1 : 0,
				            m_static_input ? 1 : 0,
				            static_cast<unsigned long long>(m_static_frame_draws),
				            static_cast<unsigned long long>(shaders), m_sampled_ranges_prev.size(),
				            m_static_snapshots.size());
				std::fflush(stdout);
			}
		}
		m_static_frame_draws = 0;
		return;
	}
	if (DepthCacheMode() >= 2) {
		const bool enough = m_depth_hash_current.size() >= 200;
		if (enough) {
			m_depth_hash_prev = std::move(m_depth_hash_current);
		} else {
			m_depth_hash_prev.clear();
		}
		m_depth_hash_current.clear();
		m_depth_hash_skip_disabled = false;
		if (DepthCacheLogEnabled()) {
			static uint32_t log_frame = 0;
			if ((++log_frame % 60u) == 0) {
				std::printf("DEPTH_HASH frame=%u prev=%zu skipped=%u rendered=%u\n", log_frame,
				            m_depth_hash_prev.size(), m_depth_hash_skip_count,
				            m_depth_hash_render_count);
				std::fflush(stdout);
			}
		}
		m_depth_hash_skip_count   = 0;
		m_depth_hash_render_count = 0;
		return;
	}
	if (DepthCacheLogEnabled()) {
		static bool logged = false;
		if (!logged) {
			logged = true;
			std::printf("DEPTH_CACHE active\n");
			std::fflush(stdout);
		}
	}
	const auto shaders = Common::DebugCounters::Get(Common::DebugCounters::Counter::ShadersCompiled);
	const bool stable  = shaders == m_depth_last_shaders;
	m_depth_last_shaders = shaders;
	const bool baseline = stable && !m_depth_uncacheable;
	bool       promote  = baseline;
	uint32_t   total    = 0;
	for (uint32_t i = 0; i < static_cast<uint32_t>(m_depth_frame.size()); i++) {
		auto& pass = m_depth_frame[i];
		if (!pass.cacheable) {
			// A pass with a placeholder is kept as a non-replayable entry; its neighbours can
			// still be promoted and replayed.
			m_depth_reason_notcacheable++;
		} else if (!pass.target_id) {
			pass.cacheable = false;
			m_depth_reason_no_target++;
		} else if (pass.target_version == 0 && !pass.draws.empty()) {
			pass.target_version = m_context.GetTextureCache().GetImage(pass.target_id).Version();
		}
		if (pass.cacheable && pass.target_id && pass.target_version != 0 &&
		    m_context.GetTextureCache().GetImage(pass.target_id).Version() != pass.target_version) {
			// Sampling a depth image refreshes cache bookkeeping and bumps its version even
			// though the pixels did not change. Keep the pass; a real colour-pass write is
			// caught by the attachment-write path below instead.
			m_depth_reason_version++;
		}
		total += static_cast<uint32_t>(pass.draws.size());
		if (pass.skipped) {
			if (i >= m_depth_cache.size() || pass.draws.size() < m_depth_cache[i].draws.size()) {
				promote = false;
			}
		}
	}
	if (m_depth_cache_ready && m_depth_frame.size() != m_depth_cache.size()) {
		promote = false;
	}
	const bool last_promote = promote && total >= 200;
	if (last_promote) {
		m_depth_cache.swap(m_depth_frame);
		if (m_depth_warmup < 1000) {
			m_depth_warmup++;
		}
		m_depth_cache_ready = m_depth_warmup >= 30;
	} else if (baseline && total >= 200) {
		// Keep the previous baseline instead of promoting a frame with a stale pass. The
		// per-pass checks reject only the images that changed.
		if (m_depth_warmup < 1000) {
			m_depth_warmup++;
		}
		m_depth_cache_ready = m_depth_warmup >= 30;
	} else {
		m_depth_cache.clear();
		m_depth_cache_ready = false;
		m_depth_warmup      = 0;
	}
	if (DepthCacheLogEnabled()) {
		static uint32_t log_frame = 0;
		if ((++log_frame % 60u) == 0) {
			std::printf("DEPTH_CACHE frame=%u ready=%d warm=%u stable=%d unc=%d total=%u "
			            "draws=%u skipped=%u state=%u range=%u target=%u pass=%u meta=%u "
			            "stencil=%u writes=%u version=%u ncache=%u cachedinv=%u compared=%u "
			            "stparts=%u/%u/%u/%u/%u "
			            "passes=%zu cachepasses=%zu promote=%d try=%.2fms rec=%.2fms "
			            "end=%.2fms\n",
			            log_frame, m_depth_cache_ready ? 1 : 0, m_depth_warmup,
			            stable ? 1 : 0, m_depth_uncacheable ? 1 : 0, total,
			            m_depth_frame_draws, m_depth_skip_count, m_depth_reason_state,
			            m_depth_reason_range, m_depth_reason_target, m_depth_reason_pass,
			            m_depth_reason_meta, m_depth_reason_stencil, m_depth_reason_writes,
			            m_depth_reason_version, m_depth_reason_notcacheable,
			            m_depth_reason_cached_invalid, m_depth_reason_compared,
			            m_depth_reason_state_ctx, m_depth_reason_state_cfg,
			            m_depth_reason_state_shader, m_depth_reason_state_args,
			            m_depth_reason_state_line, m_depth_frame.size(), m_depth_cache.size(),
			            last_promote ? 1 : 0,
			            static_cast<double>(m_depth_try_ns) / 60.0 / 1e6,
			            static_cast<double>(m_depth_record_ns) / 60.0 / 1e6,
			            static_cast<double>(m_depth_end_ns) / 60.0 / 1e6);
			std::fflush(stdout);
		}
	}
	m_depth_frame.clear();
	m_depth_range_cache.clear();
	m_depth_range_dirty.clear();
	m_depth_try_ns    = 0;
	m_depth_record_ns = 0;
	m_depth_end_ns    = 0;
	m_depth_reason_state = m_depth_reason_range = m_depth_reason_target =
	    m_depth_reason_pass = m_depth_reason_meta = m_depth_reason_stencil =
	        m_depth_reason_writes = m_depth_reason_version = m_depth_reason_notcacheable =
	            m_depth_reason_cached_invalid = m_depth_reason_compared =
	                m_depth_reason_uncacheable = 0;
	m_depth_reason_state_ctx = m_depth_reason_state_cfg = 0;
	m_depth_reason_state_shader = m_depth_reason_state_args = m_depth_reason_state_line = 0;
	m_depth_frame_draws = 0;
	m_depth_skip_count  = 0;
	m_depth_miss_count  = 0;
	m_depth_skip_rest   = false;
	m_depth_uncacheable = false;
	m_depth_record      = false;
	m_depth_pending     = {};
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            DrawRenderState& state) {
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Programs);
	if (ShouldSkipDepthOnly(buffer)) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		return false;
	}
	m_context.GetImageHeap().BeginDraw();
	m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
	// Vertex programs beyond the host mesh limits or with unsupported call targets are not
	// compiled.
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	// An empty screen scissor rasterizes nothing in any viewport, so only vertex-stage buffer
	// writes would remain (PPSA02110 ends mip reductions with a 0x0 pass past the last level).
	const auto& vp = buffer.GetRegisters().GetScreenViewport();
	if ((vp.screen_scissor_right <= vp.screen_scissor_left ||
	     vp.screen_scissor_bottom <= vp.screen_scissor_top) &&
	    std::ranges::none_of(state.VertexStages(), HasShaderBufferWrites,
	                         &ShaderVertexInputInfo::stage)) {
		return false;
	}
	auto& targets       = m_targets;
	targets.color_count = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((state.mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, targets.colors[targets.color_count], slot);
			if (targets.colors[targets.color_count].image_id) {
				targets.color_count++;
			}
		}
	}
	ResolveRenderDepthTarget(buffer, targets.depth);
	if ((DepthCacheMode() == 3 || DepthCacheMode() == 6) && !m_static_skip &&
	    targets.depth.image_id && IsDepthOnlyDraw(buffer)) {
		if (std::ranges::find(m_depth_target_ids, targets.depth.image_id) ==
		    m_depth_target_ids.end()) {
			m_depth_target_ids.push_back(targets.depth.image_id);
		}
	}
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {targets.colors, targets.color_count};
	const auto pass_samples = targets.depth.image_id ? targets.depth.desc.info.samples
	                                                 : targets.colors[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		targets.color_count -= static_cast<uint32_t>(dropped.size());
		if (targets.depth.image_id && targets.depth.desc.info.samples != raster) {
			targets.depth = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (targets.color_count == 0 && !targets.depth.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), targets.depth, buffer, draw.index_count, 0);
		return false;
	}

	// Draw mix: depth-only passes (shadow cascades and depth prepasses) are the only GPU work
	// a quality/performance trade could remove, so count them with the other draw shapes.
	if (targets.color_count == 0 && targets.depth.image_id) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		if (SkipDepthOnlyEnabled()) {
			return false;
		}
	}
	if (targets.color_count > 1) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::MrtDraw, 0);
	}
	if (targets.depth.image_id && targets.depth.desc.info.extent.depth > 1) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::LayeredDraw, 0);
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type =
	    source.guest_element_size == 4 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, const DrawTargets& targets,
                                 uint32_t index_type_and_size, const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.indexed && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	static const RenderColorInfo none {};

	const auto& color = targets.color_count != 0 ? targets.colors[0] : none;
	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), color, targets.depth, buffer, state.ps_input_info,
		                   draw.index_count, 0);
	}
	LogDrawInputState(buffer, color, state.vertex_info[0], index_type_and_size, draw.index_count,
	                  index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                      const DrawEmitInfo& emit, const DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto&       depth = m_targets.depth;
	auto&       color = m_targets.colors[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.indexed || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    m_targets.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		buffer.EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

void RenderExecutor::ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                                         const DrawRenderState& state, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source) {
	Common::PerfProbe::Scope draw_probe(Common::PerfProbe::Slot::DrawTotal);
	m_obtain_cache.Clear();
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto  vertex_stages = state.VertexStages();
	const bool  mesh_active   = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	const bool  mesh_task     = state.vertex_info[0].stage.module->mesh_task;
	const auto& mesh          = state.vertex_info[0].mesh;
	const auto& limits        = m_context.GetGraphics().mesh_shader_properties;
	uint32_t    mesh_groups   = 0;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.indexed || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed);
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed, primitive_restart_enable);
		}
	}
	if (mesh_task) {
		EXIT_NOT_IMPLEMENTED(emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
	} else if (mesh_active) {
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	// Mesh shaders read the original guest indices in whole DWORDs by device address. Their buffer
	// exists before the descriptors bind, so creating it merges no buffer this draw binds.
	uint64_t index_begin = 0;
	uint64_t index_end   = 0;
	if (mesh_active && draw.indexed) {
		const auto size = static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size;
		index_begin     = index_source.address & ~uint64_t {3};
		index_end       = (index_source.address + size + 3u) & ~uint64_t {3};
		(void)m_context.GetBufferCache().FindBuffer(index_begin, index_end - index_begin);
	}
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	std::span<PreparedBindings* const> stages;
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Bindings);
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			PrepareBindings(state.ps_input_info.stage, bindings.pixel);
			descriptor_stages[stage_count++] = &bindings.pixel;
		}
		stages = std::span {descriptor_stages.data(), stage_count};
		PrepareGraphicsBindings(stages, std::span {m_targets.colors, m_targets.color_count});
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	vk::DeviceAddress     mesh_indices = 0;
	if (!mesh_active) {
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::VertexBuffers);
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	} else if (draw.indexed) {
		const auto [indices, offset] =
		    m_context.GetBufferCache().ObtainBuffer(index_begin, index_end - index_begin, false);
		mesh_indices = indices->BufferDeviceAddress() + offset + (index_source.address & 3u);
	}
	std::pair<Buffer*, uint64_t> args_buffer {};
	if (emit.indirect.address != 0) {
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (draw.indexed ? sizeof(vk::DrawIndexedIndirectCommand)
		                                : sizeof(vk::DrawIndirectCommand));
		args_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
		EXIT_IF((args_buffer.second & 3u) != 0);
	}
	PipelineCache::Pipeline* pipeline_ptr = nullptr;
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Pipeline);
		pipeline_ptr = &m_context.GetPipelineCache().GetGraphicsPipeline(buffer, state, m_targets);
	}
	auto&                pipeline = *pipeline_ptr;
	vk::ImageAspectFlags feedback_aspects;
	RenderState          rendering;
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::RenderTargets);
		rendering = AcquireRenderTargets(buffer, m_targets.colors, m_targets.color_count,
		                                 m_targets.depth, feedback_aspects, stages);
	}
	vk::DeviceAddress task_params = 0;
	if (mesh_task) {
		const MeshTaskParams params {
		    .records        = args_buffer.first->BufferDeviceAddress() + args_buffer.second,
		    .index_address  = mesh_indices,
		    .stride         = emit.indirect.stride,
		    .index_size     = index_source.guest_element_size,
		    .index_limit    = draw.index_count,
		    .primitive_size = mesh.InputPrimitiveSize(),
		    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
		    .max_groups     = limits.maxMeshWorkGroupCount[0],
		    .max_instances  = limits.maxMeshWorkGroupCount[1],
		    .max_total      = limits.maxMeshWorkGroupTotalCount,
		};
		auto& stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		task_params  = stream.BufferDeviceAddress() + stream.Copy(&params, sizeof(params), 8);
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::CommitBindings);
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	}
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_task) {
			draw_data[0] = static_cast<uint32_t>(task_params);
			draw_data[1] = static_cast<uint32_t>(task_params >> 32u);
		} else if (mesh_active) {
			draw_data = {draw.index_count,
			             static_cast<uint32_t>(emit.vertex_offset),
			             index_source.guest_element_size,
			             static_cast<uint32_t>(mesh_indices),
			             static_cast<uint32_t>(mesh_indices >> 32u)};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::DynamicState);
		UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), m_targets.depth,
		                           rendering);
		dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
		dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);
	}

	if (std::any_of(
	        vertex_stages.begin(), vertex_stages.end(),
	        [](const auto& stage) { return stage.stage.module->info.has_address_writes; }) ||
	    (state.ps_active && state.ps_input_info.stage.module->info.has_address_writes)) {
		m_context.GetBufferCache().SupersedeAllPublishes();
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		buffer.EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::DrawCall);
		buffer.BeginRendering(rendering);
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		auto&         occlusion_queries = m_context.GetOcclusionQueries();
		const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
		if (mesh_task) {
			vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
		} else if (mesh_active) {
			vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
		} else if (args_buffer.first == nullptr) {
			EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
		} else if (draw.indexed) {
			vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
			                              emit.indirect.count, emit.indirect.stride);
		} else {
			vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second,
			                       emit.indirect.count, emit.indirect.stride);
		}
		occlusion_queries.End(vk_buffer, occlusion_query);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);
	m_context.GetCommandScheduler().NoteRecordedWork();

	if (shader_write_stages) {
		buffer.EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	{
		Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Flush);
		m_context.GetCommandScheduler().FlushForOverlap();
	}
	ReplayDepthRecordDraw(state, emit, index_source, stages);
	Common::PerfProbe::Tick();
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU:
// quad-list draws for counts, tessellation draws for the index/SGPR prolog and mesh draws without
// room for the task payload. Indirect records carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource             source,
                                                   int32_t                      first_vertex,
                                                   uint32_t                     first_instance,
                                                   const DrawIndirectArgs&      indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance} : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

bool RenderExecutor::DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	m_static_frame_draws++;
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::DrawIndexTotal);
	Common::PerfProbe::Span  prefix(Common::PerfProbe::Slot::DrawPrefix);

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	// The shadow cascades submit thousands of depth-only draws per frame. Deciding the skip here
	// keeps the lock, the metadata checks and the vertex-shader validity test off their path.
	if (EarlyDepthSkipEnabled() && IsDepthOnlyDraw(buffer)) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		return true;
	}
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	// Skip depth-only passes before the per-draw setup, not just before the program work.
	if (ShouldSkipDepthOnly(buffer)) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	DrawRenderState state {.indirect_offsets = args.indirect.offsets,
	                       .indirect_draw    = args.indirect.address != 0};
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16: index_source.guest_element_size = 2; break;
		case Prospero::IndexType::kIndex32: index_source.guest_element_size = 4; break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {true, args.index_count, args.instance_count,
	                         reinterpret_cast<uint64_t>(args.index_addr)};
	prefix.Stop();
	if (TryReplayDepthDraw(buffer, draw)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		m_depth_record = false;
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, args.index_type_and_size, args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, index_source);
	} else {
		m_depth_record = false;
	}
	ResetBindings();
	return emit.has_value();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	m_static_frame_draws++;
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::DrawIndexTotal);

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	// As in DrawIndex: the depth-only skip is decided before the lock and the metadata work.
	if (EarlyDepthSkipEnabled() && IsDepthOnlyDraw(buffer)) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		return true;
	}
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	// Skip depth-only passes before the per-draw setup, not just before the program work.
	if (ShouldSkipDepthOnly(buffer)) {
		Common::PerfProbe::Add(Common::PerfProbe::Slot::DepthOnlyDraw, 0);
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {false, args.vertex_count, args.instance_count, 0};

	DrawRenderState state {.indirect_offsets = args.indirect.offsets,
	                       .indirect_draw    = args.indirect.address != 0};
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (TryReplayDepthDraw(buffer, draw)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		m_depth_record = false;
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, {});
	} else {
		m_depth_record = false;
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, 1, true, true);
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
