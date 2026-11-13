#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_bound_pipeline       = nullptr;
	m_bound_vertex_count   = 0;
	m_bound_index_buffer   = nullptr;
	m_bound_index_offset   = 0;
	m_bound_index_type     = vk::IndexType::eUint16;
}

void CommandBuffer::BindPipelineCached(vk::Pipeline pipeline) {
	if (m_bound_pipeline == pipeline) {
		return;
	}
	m_bound_pipeline = pipeline;
	Handle().bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
}

void CommandBuffer::BindVertexBuffersCached(uint32_t count, const vk::Buffer* buffers,
                                            const uint64_t* offsets, const uint64_t* sizes) {
	if (count == m_bound_vertex_count && count <= m_bound_vertex_buffers.size() &&
	    std::memcmp(buffers, m_bound_vertex_buffers.data(), count * sizeof(vk::Buffer)) == 0 &&
	    std::memcmp(offsets, m_bound_vertex_offsets.data(), count * sizeof(uint64_t)) == 0 &&
	    std::memcmp(sizes, m_bound_vertex_sizes.data(), count * sizeof(uint64_t)) == 0) {
		return;
	}
	m_bound_vertex_count = std::min<uint32_t>(count, m_bound_vertex_buffers.size());
	if (m_bound_vertex_count != 0) {
		std::memcpy(m_bound_vertex_buffers.data(), buffers,
		            m_bound_vertex_count * sizeof(vk::Buffer));
		std::memcpy(m_bound_vertex_offsets.data(), offsets,
		            m_bound_vertex_count * sizeof(uint64_t));
		std::memcpy(m_bound_vertex_sizes.data(), sizes, m_bound_vertex_count * sizeof(uint64_t));
	}
	Handle().bindVertexBuffers2(0, count, buffers, offsets, sizes, nullptr);
}

void CommandBuffer::BindIndexBufferCached(vk::Buffer buffer, uint64_t offset, vk::IndexType type) {
	if (m_bound_index_buffer == buffer && m_bound_index_offset == offset &&
	    m_bound_index_type == type) {
		return;
	}
	m_bound_index_buffer = buffer;
	m_bound_index_offset = offset;
	m_bound_index_type   = type;
	Handle().bindIndexBuffer(buffer, offset, type);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering = false;
}

} // namespace Libs::Graphics
