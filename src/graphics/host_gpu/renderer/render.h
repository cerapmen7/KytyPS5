#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/dlss.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
struct DepthRenderTarget;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
};

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return m_buffer == nullptr; }

	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const noexcept { return m_buffer; }
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }
	// Hash of the PM4 packets that set up the next draw, supplied by the command processor.
	void SetCommandHash(uint64_t hash) noexcept { m_command_hash = hash; }
	[[nodiscard]] uint64_t GetCommandHash() const noexcept { return m_command_hash; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer = nullptr;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers    = nullptr;
	HW::UserConfig*     m_user_config  = nullptr;
	HW::Shader*         m_shaders      = nullptr;
	uint64_t            m_command_hash = 0;

	friend class CommandScheduler;
	friend struct CommandBufferTestAccess;
};

namespace Pssr {
struct Request;
}

class Buffer;

// Reuses one ObtainBuffer result for repeated (address, size, flags) within a single draw.
// The guest can rewrite the memory between draws; Clear() at each draw boundary keeps it safe.
struct BufferObtainCache {
	struct Entry {
		uint64_t address   = 0;
		uint64_t size      = 0;
		BufferId id        = {};
		bool     written   = false;
		bool     formatted = false;
		Buffer*  buffer    = nullptr;
		uint64_t offset    = 0;
	};

	static constexpr uint32_t Capacity = 64;
	std::array<Entry, Capacity> entries {};
	uint32_t                    count = 0;

	void Clear() noexcept { count = 0; }

	[[nodiscard]] bool Find(uint64_t address, uint64_t size, BufferId id, bool written,
	                        bool formatted, Buffer*& buffer, uint64_t& offset) noexcept {
		for (uint32_t i = 0; i < count; i++) {
			const auto& entry = entries[i];
			if (entry.address == address && entry.size == size && entry.id == id &&
			    entry.written == written && entry.formatted == formatted) {
				buffer = entry.buffer;
				offset = entry.offset;
				return true;
			}
		}
		return false;
	}

	void Add(uint64_t address, uint64_t size, BufferId id, bool written, bool formatted,
	         Buffer* buffer, uint64_t offset) noexcept {
		if (count >= Capacity) {
			return;
		}
		entries[count++] = {address, size, id, written, formatted, buffer, offset};
	}
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void                           NoteSampledImage(ImageId id);
	void                           RestoreSnapshot(ImageId id);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	// The PM4 front-end can drop a depth-only draw before it builds DrawIndexArgs. The executor's
	// own check still runs, so a draw that survives this one is skipped where it was before.
	[[nodiscard]] bool ShouldSkipDepthOnly(CommandBuffer& buffer);
	// A frame boundary for the depth-pass replay cache. Called once per guest flip, after the
	// frame's draws were recorded. Keeps the cached shadow/depth passes only while every draw
	// of the frame had identical inputs; otherwise the next frame is rendered from scratch.
	void DepthCacheEndFrame();

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                    const DrawEmitInfo& emit, const DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          DrawRenderState& state);
	void               ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                       const DrawRenderState& state, const DrawEmitInfo& emit,
	                                       const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool        Upscale(vk::CommandBuffer command, const Pssr::Request& request,
	                                  Dlss::Inputs& inputs);

	// DLSS history of a PSSR context; a null feature is one DLSS refused or failed.
	struct DlssContext {
		uint64_t       context;
		Dlss::Creation creation;
		Dlss::Feature* feature;
	};

	// GET_LOD_STATS counters since their last reset. With --lod-stats a bound T# with counting
	// enabled counts as sampled at its BASE_LEVEL, clamped when its MIN_LOD lies above that level.
	struct MipStatsCounter {
		uint8_t finest_level = 15; // no samples
		bool    clamped      = false;
		bool    unclamped    = false;
	};

	RenderContext&                        m_context;
	bool                                  m_lod_stats = Config::LodStatsEnabled();
	GraphicsBindings                     m_graphics_bindings;
	// The current draw's targets; PrepareDrawRenderState resolves them again for each draw.
	DrawTargets                           m_targets;
	PreparedBindings                     m_compute_bindings;
	BufferObtainCache                    m_obtain_cache;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	std::vector<DlssContext>              m_dlss_contexts;
	std::array<MipStatsCounter, 256>      m_mip_stats {};

	// Depth-pass replay cache. A shadow/depth pass whose every draw repeats with the same
	// registers and the same guest buffer contents can keep the target it produced last frame.
	// The cached pixels are therefore identical to a full re-render; a changed input falls back
	// to rendering and refreshes the cache.
	struct DepthCacheRange {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t hash    = 0;
	};
	struct DepthCacheImage {
		ImageId  id      = {};
		uint64_t version = 0;
	};
	struct DepthCacheDraw {
		// Fixed storage: thousands of draws are recorded every frame, and per-draw vectors
		// would allocate tens of thousands of small blocks and dominate the frame.
		static constexpr uint32_t MaxRanges = 32;
		static constexpr uint32_t MaxImages = 8;
		uint64_t                              state_hash  = 0;
		uint64_t                              state_ctx   = 0;
		uint64_t                              state_cfg   = 0;
		uint64_t                              state_shader = 0;
		uint64_t                              state_args  = 0;
		uint64_t                              state_line  = 0;
		uint32_t                              range_count = 0;
		uint32_t                              image_count = 0;
		// A draw whose GPU-written inputs cannot be inspected is a placeholder that never
		// matches: it renders every frame while its neighbours still replay.
		bool                                  cacheable   = true;
		std::array<DepthCacheRange, MaxRanges> ranges {};
		std::array<DepthCacheImage, MaxImages> images {};
	};
	struct DepthCachePass {
		uint64_t                 target_hash    = 0;
		ImageId                  target_id      = {};
		uint64_t                 target_version = 0;
		std::vector<DepthCacheDraw> draws;
		bool                     cacheable = true;
		bool                     skipped   = false;
		bool                     miss      = false;
		bool                     skip_rest = false;
	};

	[[nodiscard]] bool TryReplayDepthDraw(CommandBuffer& buffer, const DrawCallInfo& draw);
	void ReplayDepthRecordDraw(const DrawRenderState& state, const DrawEmitInfo& emit,
	                           const DrawIndexBufferSource& index_source,
	                           std::span<PreparedBindings* const> stages);
	[[nodiscard]] static uint64_t DepthStateHash(const CommandBuffer& buffer,
	                                             const DrawCallInfo& draw,
	                                             DepthCacheDraw*    parts);
	[[nodiscard]] static uint64_t DepthTargetHash(const HW::DepthRenderTarget& target, bool clear);
	[[nodiscard]] uint64_t DepthRangeHash(uint64_t address, uint64_t size,
	                                      const DepthCacheDraw* previous, uint32_t index);
	[[nodiscard]] static bool DepthCacheEnabled();
	[[nodiscard]] static int  DepthCacheMode();
	[[nodiscard]] static uint64_t StaticSnapshotKey(const ImageInfo& info);

	std::vector<DepthCachePass> m_depth_cache;
	std::vector<DepthCachePass> m_depth_frame;
	std::unordered_map<uint64_t, DepthCacheRange> m_depth_range_cache;
	std::unordered_set<uint64_t>                  m_depth_range_dirty;
	DepthCacheDraw               m_depth_pending {};
	uint64_t                     m_depth_pending_hash = 0;
	uint32_t                     m_depth_pending_pass = 0;
	uint32_t                     m_depth_warmup       = 0;
	uint32_t                     m_depth_frame_draws  = 0;
	uint32_t                     m_depth_skip_count   = 0;
	uint32_t                     m_depth_miss_count   = 0;
	uint32_t                     m_depth_reason_state = 0;
	uint32_t                     m_depth_reason_state_ctx = 0;
	uint32_t                     m_depth_reason_state_cfg = 0;
	uint32_t                     m_depth_reason_state_shader = 0;
	uint32_t                     m_depth_reason_state_args = 0;
	uint32_t                     m_depth_reason_state_line = 0;
	uint32_t                     m_depth_reason_range = 0;
	uint32_t                     m_depth_reason_target = 0;
	uint32_t                     m_depth_reason_pass   = 0;
	uint32_t                     m_depth_reason_meta   = 0;
	uint32_t                     m_depth_reason_stencil = 0;
	uint32_t                     m_depth_reason_writes  = 0;
	uint32_t                     m_depth_reason_version = 0;
	uint32_t                     m_depth_reason_notcacheable = 0;
	uint32_t                     m_depth_reason_cached_invalid = 0;
	uint32_t                     m_depth_reason_compared = 0;
	uint32_t                     m_depth_reason_no_target = 0;
	uint32_t                     m_depth_reason_uncacheable = 0;
	uint64_t                     m_depth_last_shaders = 0;
	uint64_t                     m_depth_try_ns       = 0;
	uint64_t                     m_depth_record_ns    = 0;
	uint64_t                     m_depth_end_ns       = 0;
	bool                         m_depth_record       = false;
	bool                         m_depth_cache_ready  = false;
	bool                         m_depth_skip_rest    = false;
	bool                         m_depth_uncacheable  = false;
	std::vector<uint64_t>        m_depth_hash_prev;
	std::vector<uint64_t>        m_depth_hash_current;
	bool                         m_depth_hash_skip_disabled = false;
	uint32_t                     m_depth_hash_skip_count = 0;
	uint32_t                     m_depth_hash_render_count = 0;
	// Mode 3: reuse the last fully rendered shadow/depth targets while the scene is idle.
	bool                         m_static_skip = false;
	bool                         m_static_input = false;
	uint32_t                     m_static_idle = 0;
	uint32_t                     m_static_refresh = 0;
	uint64_t                     m_static_frame_draws = 0;
	uint64_t                     m_static_last_shaders = 0;
	bool                         m_static_logged = false;
	std::vector<ImageId>         m_sampled_ids;
	std::unordered_set<uint32_t> m_sampled_seen;
	std::vector<std::pair<uint64_t, uint64_t>> m_sampled_ranges_prev;
	struct StaticSnapshot {
		uint64_t                key = 0;
		std::shared_ptr<Image>  image;
	};
	std::vector<ImageId>                  m_depth_target_ids;
	std::vector<ImageId>                  m_sampled_ids_prev;
	std::vector<StaticSnapshot>           m_static_snapshots;
	std::unordered_map<uint32_t, uint64_t> m_restored_versions;
	bool                                  m_static_frame_skipped = false;
	bool                                  m_sampled_restored = false;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
