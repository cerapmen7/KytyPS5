#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/submitCadence.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	/// Creates a scheduler. Only the render scheduler asks for the submit cadence. The presenter's
	/// scheduler submits at its own frame boundaries and must never split its command buffer, so
	/// every caller states the choice.
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool submit_cadence);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush();
	void           Flush(SubmitInfo& submit);
	// Submit completed draws periodically so the GPU can start executing the frame before the
	// whole command stream has been recorded. KYTY_OVERLAP_DRAWS sets the interval (0 disables).
	void           FlushForOverlap();
	// Like Flush, but can defer the submit across KYTY_COALESCE_FLUSH end-of-slice flushes so the
	// CPU does not pay one vkQueueSubmit per guest submission. A Wait/Finish/frame boundary still
	// submits everything recorded so far.
	void           FlushCoalesced();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	/// Tells the submit cadence that a draw or dispatch went into the current command buffer.
	void NoteRecordedWork() { m_cadence.NoteWork(); }
	/// Returns whether the submit cadence is on, which makes scheduler ticks much faster than
	/// frames.
	[[nodiscard]] bool SubmitCadenceEnabled() const noexcept { return m_cadence.Enabled(); }
	/// Returns whether the PM4 loop should submit the current command buffer now. While nothing is
	/// due the inline gate reads at most the clock, once per boundary stride, so the GPU progress
	/// is read only after the time gate passed.
	[[nodiscard]] bool TakeCadenceSubmit() {
		return m_cadence.DueAtBoundary() && TakeDueCadenceSubmit();
	}
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 32;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	friend struct CommandSchedulerTestAccess;

	/// Refreshes the GPU progress and asks the policy whether a due submit may go ahead, which it
	/// refuses while the GPU lags. Also feeds the held back counter.
	[[nodiscard]] bool TakeDueCadenceSubmit();

	/// Returns how many submitted command buffers the GPU has not finished. It saturates at zero,
	/// since the known GPU tick can pass the last submitted tick between two reads.
	[[nodiscard]] static uint64_t GpuLag(uint64_t current_tick, uint64_t known_gpu_tick) noexcept;

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	// Mirrors !m_pending_operations.empty() so the per-draw PopPendingOperations can skip
	// the semaphore refresh and the mutex when nothing is deferred (the common case).
	std::atomic<bool>            m_has_pending_operations {false};
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	uint64_t                     m_priority_active_tick = 0; // 0 while none runs
	uint32_t                     m_overlap_draws        = 0;
	OperationState               m_operation_state      = OperationState::Open;
	SubmitCadence                m_cadence;
	uint32_t                     m_coalesced_flushes = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
