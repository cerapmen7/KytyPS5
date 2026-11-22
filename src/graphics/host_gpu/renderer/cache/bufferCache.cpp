#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/perfProbe.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
// All four can be tuned for lab A/B without a rebuild.
template <typename T>
T LabEnv(const char* name, T fallback) {
	const auto* const value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return static_cast<T>(std::strtoull(value, nullptr, 10));
}
constexpr uint64_t DefaultDownloadWindowSize = 512 * 1024;
// Packed download ranges start on their own cache line, as in shadPS4.
constexpr uint64_t DownloadCopyAlignment = 64;

constexpr uint64_t DefaultHotFlips      = 16;
constexpr size_t   DefaultHotPagesMax   = 8192;
constexpr uint64_t HotNotePages         = 16;
constexpr uint64_t DefaultPublishBudget = 2 * MiB;

void RecordDownloadCopies(vk::CommandBuffer native, const Buffer& source, const Buffer& target,
                          std::span<const vk::BufferCopy> copies, uint64_t offset, uint64_t size) {
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = source.Handle();
	before.offset              = 0;
	before.size                = source.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(source.Handle(), target.Handle(), static_cast<uint32_t>(copies.size()),
	                  copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = target.Handle();
	after.offset        = offset;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
}

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

/// Calls visit(address, size) for each part of the range that claimed does not cover, in address
/// order.
template <class Visit>
void ForEachUnclaimed(const RangeSet& claimed, uint64_t address, uint64_t size, Visit&& visit) {
	const uint64_t end    = address + size;
	uint64_t       cursor = address;
	claimed.ForEachInRange(address, size, [&](uint64_t begin, uint64_t last) {
		if (begin > cursor) {
			visit(cursor, begin - cursor);
		}
		cursor = last;
	});
	if (cursor < end) {
		visit(cursor, end - cursor);
	}
}

/// Returns the tracker pages a batch already downloads. The main window is rounded out to whole
/// pages, because its caller unmarks it only after the batch and a recent write that shares one of
/// its pages must not collect that page a second time.
template <class Batch>
RangeSet ClaimedDownloadPages(const Batch& batch, uint64_t vaddr, uint64_t size) {
	RangeSet   claimed;
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	claimed.Add(begin, Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - begin);
	for (size_t index = 1; index < batch.size(); ++index) {
		claimed.Add(batch[index].unmark_address, batch[index].unmark_size);
	}
	return claimed;
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		{
			std::lock_guard lock(m_bda_sync_mutex);
			m_dma_accessed.ForEachInRange(
			    buffer.CpuAddress(), buffer.Size(), [&](uint64_t begin, uint64_t end) {
				    for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
					    addresses[buffer.Offset(address) >> CACHING_PAGEBITS] += BDA_ACCESSED;
				    }
			    });
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

/// Takes the GPU-dirty ranges of a buffer window out of the tracking sets and lists the copies
/// that download them. The result has no copies when nothing in the window was GPU-dirty.
BufferCache::PendingDownload BufferCache::CollectDownloadCopies(const Buffer& buffer,
                                                                uint64_t vaddr, uint64_t size) {
	PendingDownload pending;
	pending.buffer_address = buffer.CpuAddress();
	const bool                                 serve = ServeCpuReadsEnabled();
	std::vector<std::pair<uint64_t, uint64_t>> cleared;
	uint64_t                                   serve_begin = UINT64_MAX;
	uint64_t                                   serve_end   = 0;
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    pending.copies.emplace_back(start - pending.buffer_address, pending.total_size,
			                                end - start);
			    // Start each packed range on its own 64-byte cache line.
			    pending.total_size += Common::AlignUp(end - start, DownloadCopyAlignment);
			    serve_begin = std::min(serve_begin, start);
			    serve_end   = std::max(serve_end, end);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    if (serve) {
			    cleared.emplace_back(address, bytes);
		    }
		    ForgetColorMetadataChecks(address, bytes);
		    SupersedePublishes(address, bytes);
	    });
	if (serve && !cleared.empty()) {
		// The collected bytes leave the dirty mirror and enter the in-flight list in one step,
		// so a concurrent served read never sees them as clean before they are published.
		std::unique_lock lock(m_serve_mutex);
		if (!pending.copies.empty()) {
			pending.serve_id = m_serve_next_id++;
			m_serve_inflight.push_back({serve_begin, serve_end, pending.serve_id});
		}
		for (const auto& [address, bytes]: cleared) {
			m_serve_dirty.Subtract(address, bytes);
		}
	}
	return pending;
}

bool BufferCache::ServeCpuReadsEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SERVE_CPU_READS");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

void BufferCache::ServeMarkDirty(uint64_t vaddr, uint64_t size) {
	if (!ServeCpuReadsEnabled()) {
		return;
	}
	std::unique_lock lock(m_serve_mutex);
	m_serve_dirty.Add(vaddr, size);
}

void BufferCache::ServeClearDirty(uint64_t vaddr, uint64_t size) {
	if (!ServeCpuReadsEnabled()) {
		return;
	}
	std::unique_lock lock(m_serve_mutex);
	m_serve_dirty.Subtract(vaddr, size);
}

void BufferCache::ServeEndInflight(uint64_t id) {
	if (id == 0) {
		return;
	}
	std::unique_lock lock(m_serve_mutex);
	std::erase_if(m_serve_inflight, [id](const ServeInflight& entry) { return entry.id == id; });
}

bool BufferCache::CanServeCpuRead(uint64_t vaddr, uint64_t size) const {
	if (!ServeCpuReadsEnabled() || size == 0 || size > UINT64_MAX - vaddr) {
		return false;
	}
	constexpr uint64_t HostPage = 0x1000;
	const auto         end      = vaddr + size;
	std::shared_lock   lock(m_serve_mutex);
	if (m_serve_dirty.Intersects(vaddr, size)) {
		return false;
	}
	for (const auto& entry: m_serve_inflight) {
		if (entry.begin < end && vaddr < entry.end) {
			return false;
		}
	}
	// Only reads that fault because of GPU-written bytes on their page are served; any other
	// protected page keeps the normal fault path.
	const auto first_page = Common::AlignDown(vaddr, HostPage);
	const auto last_page  = Common::AlignDown(end - 1, HostPage);
	return m_serve_dirty.Intersects(first_page, HostPage) ||
	       (last_page != first_page && m_serve_dirty.Intersects(last_page, HostPage));
}

/// Reserves download memory for all the pending copies at once, from the stream buffer or from a
/// temporary buffer that the pending downloads then share. One reservation for the whole batch
/// lets the ring wait as a single download would, which is safe because nothing of the batch is
/// recorded yet. Reserving per download could wrap onto an earlier download's unpublished bytes.
void BufferCache::ReserveDownloadMemory(std::span<PendingDownload* const> items) {
	uint64_t total_size = 0;
	for (const auto* item: items) {
		// A second reservation would orphan the first one's temporary buffer and offsets.
		EXIT_IF(item->mapped != nullptr);
		total_size += item->total_size;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	auto [mapped, offset] = m_download_buffer.Map(total_size, DownloadCopyAlignment);
	std::shared_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped    = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	// Each download is a slice of the reservation. Every total_size is already a multiple of
	// DownloadCopyAlignment, so each slice starts on its own cache line.
	uint64_t slice = 0;
	for (auto* item: items) {
		item->temporary = temporary;
		item->mapped    = mapped + slice;
		item->offset    = offset + slice;
		for (auto& copy: item->copies) {
			copy.dstOffset += item->offset;
		}
		slice += item->total_size;
	}
}

/// Records the reserved copies into the current command, wrapped in the barriers that make the
/// GPU writes visible to the transfer and the downloaded bytes visible to the host.
void BufferCache::RecordDownload(const Buffer& buffer, const PendingDownload& pending) {
	EXIT_IF(pending.mapped == nullptr);
	const auto& download = pending.temporary ? *pending.temporary : m_download_buffer;

	auto& command = m_scheduler.Current();
	command.EndRendering();
	RecordDownloadCopies(command.Handle(), buffer, download, pending.copies, pending.offset,
	                     pending.total_size);
}

/// Copies finished download bytes into guest memory. The GPU must have completed the copies.
void BufferCache::PublishDownload(const PendingDownload& pending) {
	(pending.temporary ? *pending.temporary : m_download_buffer)
	    .Invalidate(pending.offset, pending.total_size);
	for (const auto& copy: pending.copies) {
		Libs::LibKernel::Memory::WriteBacking(pending.buffer_address + copy.srcOffset,
		                                      pending.mapped + (copy.dstOffset - pending.offset),
		                                      copy.size);
	}
	ServeEndInflight(pending.serve_id);
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Readback);
	auto pending = CollectDownloadCopies(buffer, vaddr, size);
	if (pending.copies.empty()) {
		return false;
	}
	PendingDownload* const items[] = {&pending};
	ReserveDownloadMemory(items);
	RecordDownload(buffer, pending);
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(
		    [this, pending = std::move(pending)] { PublishDownload(pending); });
	} else {
		const auto tick = m_scheduler.CurrentTick();
		{
			Common::PerfProbe::CoarseScope wait(Common::PerfProbe::Slot::WaitDownload);
			m_scheduler.Wait(tick);
		}
		m_scheduler.WaitPriorityOperations(tick);
		PublishDownload(pending);
	}
	return true;
}

/// Finds the buffer that owns a hot entry. The page table is read directly because FindBuffer
/// would create a buffer on a miss, and a hot entry may outlive its buffer.
BufferId BufferCache::FindHotBuffer(uint64_t address, uint64_t size) {
	const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
	if (owner == nullptr || IsBufferInvalid(*owner) ||
	    !m_slot_buffers[*owner].IsInBounds(address, size)) {
		return {};
	}
	return *owner;
}

/// Returns an upper bound on the bytes that downloading a range would add to the ring. It counts
/// every GPU-written range inside it, which is at least what the page tracker lets through.
uint64_t BufferCache::EstimateDownloadBytes(uint64_t address, uint64_t size) const {
	uint64_t bytes = 0;
	m_gpu_modified_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
		bytes += Common::AlignUp(end - start, DownloadCopyAlignment);
	});
	return bytes;
}

/// Adds the GPU-written bytes of recently read-back buffers to a batch, so the wait that the main
/// download needs anyway also clears their next fault. A buffer is skipped when it is the main
/// buffer, was already added, no longer exists, or would cost more than what is left of budget.
/// The main-buffer skip is needed for correctness. The main buffer is nearly always hot, and
/// collecting it again would revisit window pages the caller has not unmarked yet, which breaks the
/// debug invariant in ValidateGpuDirtyPages and downloads bytes outside the main window. The
/// duplicate skip only saves a repeated estimate, since the first collect already removed the
/// buffer's ranges.
void BufferCache::CollectHotDownloads(BufferId main_id, uint64_t& budget,
                                      std::vector<BatchedDownload>& batch) {
	std::vector<BufferId> seen;
	seen.reserve(MaxHotReadbackEntries + 1);
	seen.push_back(main_id);
	m_hot_readbacks.ForEachHot(m_scheduler.CurrentTick(), [&](uint64_t address, uint64_t size) {
		// Merged buffers can leave two entries that resolve to the same buffer.
		const auto id = FindHotBuffer(address, size);
		if (!id || std::ranges::find(seen, id) != seen.end()) {
			return;
		}
		seen.push_back(id);
		auto&      hot      = m_slot_buffers[id];
		const auto estimate = EstimateDownloadBytes(hot.CpuAddress(), hot.Size());
		if (estimate == 0 || estimate > budget) {
			return;
		}
		const auto hot_address = hot.CpuAddress();
		const auto hot_size    = hot.Size();
		auto       pending     = CollectDownloadCopies(hot, hot_address, hot_size);
		if (pending.copies.empty()) {
			return;
		}
		budget -= std::min(budget, pending.total_size);
		batch.push_back({&hot, std::move(pending), hot_address, hot_size});
	});
}

/// Walks the recent write ring newest first and collects each finished span that is not claimed
/// yet. Every visited span is claimed afterwards, so an older entry under it is clipped to what it
/// adds. Pending spans are claimed without being collected, which keeps an older finished write
/// that the GPU is rewriting out of the download. Claims are whole tracker pages, so a pending
/// write also holds back finished bytes that share its pages. That only leaves them for their own
/// fault. The walk ends once the budget is spent.
void BufferCache::CollectRecentWrites(RangeSet& claimed, uint64_t& budget,
                                      std::vector<BatchedDownload>& batch) {
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto known_tick = master.KnownGpuTick();
	if (known_tick == 0) {
		return;
	}
	// An entry carries the tick that was current when its buffer was bound, and a submit during
	// draw setup can put the write itself in the next tick. Only ticks before the known one are
	// treated as finished, so the entry's own write is done. That says nothing about a later
	// write the ring never records, such as a large buffer or an image, that is bound but not yet
	// recorded. Like the hot buffers, this relies on no readback running between such a bind and
	// its recording.
	m_recent_writes.ForEachNewest(known_tick - 1, [&](uint64_t address, uint64_t size,
	                                                  bool finished) {
		if (finished) {
			ForEachUnclaimed(claimed, address, size, [&](uint64_t piece, uint64_t piece_size) {
				CollectRecentPiece(piece, piece_size, budget, batch);
			});
		}
		claimed.Add(address, size);
		return budget > 0;
	});
}

/// Collects the GPU-written bytes of one unclaimed piece of a recent write and counts it for the
/// overlay. A piece is skipped when no single buffer holds it or when it would cost more than
/// budget.
void BufferCache::CollectRecentPiece(uint64_t address, uint64_t size, uint64_t& budget,
                                     std::vector<BatchedDownload>& batch) {
	const auto id = FindHotBuffer(address, size);
	if (!id) {
		return;
	}
	const auto estimate = EstimateDownloadBytes(address, size);
	if (estimate == 0 || estimate > budget) {
		return;
	}
	auto& buffer  = m_slot_buffers[id];
	auto  pending = CollectDownloadCopies(buffer, address, size);
	if (pending.copies.empty()) {
		return;
	}
	budget -= std::min(budget, pending.total_size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferCoDownloads);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferCoDownloadBytes,
	                           pending.total_size);
	batch.push_back({&buffer, std::move(pending), address, size});
}

/// Downloads the main window, the GPU-written bytes of hot buffers and the finished recent small
/// writes with one GPU wait. The extras are marked clean here, and the caller does that for the
/// main window. Nothing is marked CPU-modified, since only a faulting write does that. Returns
/// false when the window had nothing.
bool BufferCache::DownloadWithHotBuffers(BufferId main_id, Buffer& buffer, uint64_t vaddr,
                                         uint64_t size) {
	auto main_pending = CollectDownloadCopies(buffer, vaddr, size);
	if (main_pending.copies.empty()) {
		return false;
	}
	std::vector<BatchedDownload> batch;
	batch.reserve(MaxHotReadbackEntries + 1);
	batch.push_back({&buffer, std::move(main_pending)});
	uint64_t budget = MaxHotReadbackBytes;
	CollectHotDownloads(main_id, budget, batch);
	auto claimed = ClaimedDownloadPages(batch, vaddr, size);
	CollectRecentWrites(claimed, budget, batch);
	// The pointers are taken once the batch has stopped growing, so none can be invalidated.
	std::vector<PendingDownload*> pendings;
	pendings.reserve(batch.size());
	for (auto& item: batch) {
		pendings.push_back(&item.pending);
	}
	ReserveDownloadMemory(pendings);
	for (const auto& item: batch) {
		RecordDownload(*item.buffer, item.pending);
	}
	const auto tick = m_scheduler.CurrentTick();
	{
		Common::PerfProbe::CoarseScope wait(Common::PerfProbe::Slot::WaitDownload);
		m_scheduler.Wait(tick);
	}
	m_scheduler.WaitPriorityOperations(tick);
	for (const auto& item: batch) {
		PublishDownload(item.pending);
	}
	for (size_t index = 1; index < batch.size(); ++index) {
		UnmarkCleanPages(batch[index].unmark_address, batch[index].unmark_size);
	}
	return true;
}

bool BufferCache::AsyncReadbacksEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_READBACK");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

std::shared_ptr<BufferCache::AsyncReadback> BufferCache::FindAsyncReadback(uint64_t vaddr,
                                                                           uint64_t size) const {
	// Whole tracker pages: a page is unprotected as a unit.
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (const auto& state: m_async_readbacks) {
		const auto begin = Common::AlignDown(state->start, TRACKER_PAGE_SIZE);
		const auto end   = Common::AlignUp(state->start + state->size, TRACKER_PAGE_SIZE);
		if (begin < last && first < end) {
			return state;
		}
	}
	return nullptr;
}

bool BufferCache::AsyncReadbackIntersects(uint64_t vaddr, uint64_t size) const {
	return !m_async_readbacks.empty() && FindAsyncReadback(vaddr, size) != nullptr;
}

void BufferCache::FinishAsyncReadback(const std::shared_ptr<AsyncReadback>& state) {
	{
		std::unique_lock lock(state->mutex);
		state->published.wait(lock, [&] { return state->done; });
	}
	const auto it = std::ranges::find(m_async_readbacks, state);
	if (it == m_async_readbacks.end()) {
		return;
	}
	m_async_readbacks.erase(it);
	UnmarkCleanPages(state->start, state->size);
}

void BufferCache::UnmarkCleanPages(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	const auto first     = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last      = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	uint64_t   run_start = 0;
	uint64_t   run_size  = 0;
	const auto flush_run = [&] {
		if (run_size != 0) {
			m_memory_tracker.UnmarkRegionAsGpuModified(run_start, run_size);
			run_size = 0;
		}
	};
	// Consecutive clean pages are unprotected together, as the single-range unmark did before.
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		const bool clean = !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
		                   !AsyncReadbackIntersects(page, TRACKER_PAGE_SIZE);
		if (!clean) {
			flush_run();
			continue;
		}
		if (run_size == 0) {
			run_start = page;
		}
		run_size += TRACKER_PAGE_SIZE;
	}
	flush_run();
}

bool BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write) {
	std::shared_ptr<AsyncReadback> state;
	bool                           owner   = false;
	bool                           handled = false;
	uint64_t                       dirty   = 0;
	auto&                          gpu     = m_scheduler.Context().GetGpu();
	const uint64_t                 t0      = Common::PerfProbe::LiteEnabled() ? Common::PerfProbe::Now() : 0;
	gpu.SendCommandSync([&] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			handled = true;
			return;
		}
		// Another guest thread's download already covers this page: wait for the same publish.
		if (auto existing = FindAsyncReadback(vaddr, size)) {
			state = std::move(existing);
			return;
		}
		if (!IsRegionGpuModified(vaddr, size)) {
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
				QueueBdaSync(vaddr, size, true);
			}
			handled = true;
			return;
		}
		const auto buffer_id = FindBuffer(vaddr, size);
		auto&      buffer    = m_slot_buffers[buffer_id];
		static const uint64_t download_window =
		    LabEnv("KYTY_DOWNLOAD_WINDOW", DefaultDownloadWindowSize);
		const auto start = std::max(Common::AlignDown(vaddr, download_window), buffer.CpuAddress());
		const auto end   = std::min(std::max(start + download_window, vaddr + size),
		                            buffer.CpuAddress() + buffer.Size());
		auto pending = CollectDownloadCopies(buffer, start, end - start);
		if (pending.copies.empty()) {
			return; // Unusual tracker state: leave it to the synchronous path.
		}
		dirty = pending.total_size;
		PendingDownload* const items[] = {&pending};
		ReserveDownloadMemory(items);
		RecordDownload(buffer, pending);
		state                 = std::make_shared<AsyncReadback>();
		state->start          = start;
		state->size           = end - start;
		state->buffer_address = buffer.CpuAddress();
		state->buffer_size    = buffer.Size();
		m_async_readbacks.push_back(state);
		owner = true;
		// Queued before the flush so the operation's tick is the command buffer holding the copy.
		m_scheduler.DeferPriorityOperation([this, state, pending = std::move(pending)] {
			PublishDownload(pending);
			{
				std::lock_guard lock(state->mutex);
				state->done = true;
			}
			state->published.notify_all();
		});
		m_scheduler.Flush();
	});
	if (handled) {
		return true;
	}
	if (!state) {
		return false;
	}
	{
		Common::PerfProbe::CoarseScope wait(Common::PerfProbe::Slot::AsyncDlWait);
		std::unique_lock lock(state->mutex);
		state->published.wait(lock, [&] { return state->done; });
	}
	bool fallback = false;
	gpu.SendCommandSync([&] {
		FinishAsyncReadback(state);
		if (owner) {
			NoteHotReadback(state->buffer_address, state->buffer_size);
			NoteReadHot(vaddr, size);
		}
		if (is_write) {
			// Rewritten by the GPU meanwhile: the synchronous path downloads it again.
			if (IsRegionGpuModified(vaddr, size)) {
				fallback = true;
				return;
			}
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size, true);
		}
	});
	if (t0 != 0) {
		Common::PerfProbe::Trace("RBA %s owner=%d addr=0x%llx dirty=%llu us=%llu fallback=%d",
		                         is_write ? "w" : "r", owner ? 1 : 0,
		                         static_cast<unsigned long long>(vaddr),
		                         static_cast<unsigned long long>(dirty),
		                         static_cast<unsigned long long>((Common::PerfProbe::Now() - t0) / 1000),
		                         fallback ? 1 : 0);
	}
	return !fallback;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream,
                      LabEnv("KYTY_STREAM_MB", 256) * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 256 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache), m_batch_hot_readbacks(Config::BatchHotReadbacksEnabled()) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size, true);
}

void BufferCache::NoteHotReadback(uint64_t address, uint64_t size) {
	if (m_batch_hot_readbacks) {
		m_hot_readbacks.Note(address, size, m_scheduler.CurrentTick());
	}
}

void BufferCache::NoteRecentWrite(uint64_t address, uint64_t size) {
	if (m_batch_hot_readbacks) {
		m_recent_writes.Note(address, size, m_scheduler.CurrentTick());
	}
}

// Lab trace tag for GPU writes: 0 = shader binding, 1 = DMA fill, 2 = DMA copy (3 = copy with a
// GPU-written source).
static thread_local int g_write_source = 0;

void BufferCache::NoteWriteTick(uint64_t vaddr, uint64_t size) {
	if (!Common::PerfProbe::LiteEnabled() || size == 0) {
		return;
	}
	if (!m_write_ticks) {
		m_write_ticks = std::make_unique<WriteTickEntry[]>(WriteTickSlots);
	}
	const auto tick  = m_scheduler.CurrentTick();
	const auto first = vaddr >> 12u;
	const auto last  = std::min((vaddr + size - 1) >> 12u, first + 4095);
	for (auto page = first; page <= last; ++page) {
		m_write_ticks[page & (WriteTickSlots - 1)] = {page, tick};
	}
}

uint64_t BufferCache::MaxWriteTick(uint64_t vaddr, uint64_t size) const {
	if (!m_write_ticks || size == 0) {
		return 0;
	}
	uint64_t   tick  = 0;
	const auto first = vaddr >> 12u;
	const auto last  = std::min((vaddr + size - 1) >> 12u, first + 4095);
	for (auto page = first; page <= last; ++page) {
		const auto& entry = m_write_ticks[page & (WriteTickSlots - 1)];
		if (entry.page == page) {
			tick = std::max(tick, entry.tick);
		}
	}
	return tick;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const uint64_t fault_rip     = Common::PerfProbe::FaultRipRef();
	const bool     gpu_caller    = GuestGpu::IsGpuThread();
	const bool     render_caller = Common::PerfProbe::RenderThreadRef();
	if (!gpu_caller && AsyncReadbacksEnabled() && ReadMemoryAsync(vaddr, size, is_write)) {
		return;
	}
	const uint64_t queued_ns     = Common::PerfProbe::LiteEnabled() ? Common::PerfProbe::Now() : 0;
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, fault_rip,
	                                                gpu_caller, render_caller, queued_ns] {
		// An asynchronous download over these pages holds bytes that are no longer in the dirty
		// set: wait for its publish (this blocks the GPU thread, as every sync download does).
		while (auto inflight = FindAsyncReadback(vaddr, size)) {
			FinishAsyncReadback(inflight);
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		const auto buffer_id = FindBuffer(vaddr, size);
		auto&      buffer    = m_slot_buffers[buffer_id];
		if (IsRegionGpuModified(vaddr, size)) {
			static const uint64_t download_window =
			    LabEnv("KYTY_DOWNLOAD_WINDOW", DefaultDownloadWindowSize);
			const auto start =
			    std::max(Common::AlignDown(vaddr, download_window), buffer.CpuAddress());
			const auto end = std::min(std::max(start + download_window, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			const bool     lite       = Common::PerfProbe::LiteEnabled();
			uint64_t       trace_t0   = 0;
			uint64_t       known_tick = 0;
			uint64_t       wtick_page = 0;
			uint64_t       wtick_win  = 0;
			uint64_t       dirty      = 0;
			uint64_t       cur_tick   = 0;
			int            hit        = 0;
			char           ranges[160] {};
			if (lite) {
				auto& master = m_scheduler.GetMasterSemaphore();
				master.Refresh();
				known_tick = master.KnownGpuTick();
				cur_tick   = m_scheduler.CurrentTick();
				wtick_page = MaxWriteTick(vaddr, size);
				wtick_win  = MaxWriteTick(start, end - start);
				dirty      = EstimateDownloadBytes(start, end - start);
				// Does the access itself touch GPU-written bytes (32-byte span), or only its page?
				hit        = m_gpu_modified_ranges.Intersects(vaddr, 32) ? 1 : 0;
				size_t     pos   = 0;
				int        shown = 0;
				const auto base  = Common::AlignDown(vaddr, uint64_t {0x4000});
				m_gpu_modified_ranges.ForEachInRange(base, 0x4000, [&](uint64_t begin, uint64_t finish) {
					if (shown++ < 4 && pos < sizeof(ranges) - 40) {
						pos += static_cast<size_t>(std::snprintf(
						    ranges + pos, sizeof(ranges) - pos, "%llx-%llx,",
						    static_cast<unsigned long long>(std::max(begin, base) - base),
						    static_cast<unsigned long long>(std::min(finish, base + 0x4000) - base)));
					}
				});
				trace_t0 = Common::PerfProbe::Now();
			}
			const bool downloaded =
			    m_batch_hot_readbacks
			        ? DownloadWithHotBuffers(buffer_id, buffer, start, end - start)
			        : DownloadBufferMemory<false>(buffer, start, end - start);
			if (lite) {
				Common::PerfProbe::Trace(
				    "RB %s rt=%d hit=%d q_us=%llu rng=%s rip=0x%llx addr=0x%llx size=%llu "
				    "win=0x%llx+0x%llx buf=0x%llx+0x%llx "
				    "dirty=%llu dl=%d cur=%llu known=%llu wpage=%llu wwin=%llu us=%llu",
				    gpu_caller ? "gpu" : (is_write ? "cpuw" : "cpur"), render_caller ? 1 : 0, hit,
				    static_cast<unsigned long long>((trace_t0 - queued_ns) / 1000),
				    ranges[0] != '\0' ? ranges : "-",
				    static_cast<unsigned long long>(fault_rip),
				    static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
				    static_cast<unsigned long long>(start),
				    static_cast<unsigned long long>(end - start),
				    static_cast<unsigned long long>(buffer.CpuAddress()),
				    static_cast<unsigned long long>(buffer.Size()),
				    static_cast<unsigned long long>(dirty), downloaded ? 1 : 0,
				    static_cast<unsigned long long>(cur_tick),
				    static_cast<unsigned long long>(known_tick),
				    static_cast<unsigned long long>(wtick_page),
				    static_cast<unsigned long long>(wtick_win),
				    static_cast<unsigned long long>((Common::PerfProbe::Now() - trace_t0) / 1000));
			}
			if (downloaded) {
				UnmarkCleanPages(start, end - start);
				NoteHotReadback(buffer.CpuAddress(), buffer.Size());
				NoteReadHot(vaddr, size);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size, true);
		}
	});
}

void BufferCache::NoteReadHot(uint64_t vaddr, uint64_t size) {
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (last - first > HotNotePages * TRACKER_PAGE_SIZE) {
		return;
	}
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		if (auto it = m_hot_pages.find(page); it != m_hot_pages.end()) {
			it->second = m_hot_flip;
		} else if (m_hot_pages.size() < LabEnv("KYTY_HOT_PAGES", DefaultHotPagesMax)) {
			m_hot_pages.emplace(page, m_hot_flip);
			m_hot_ranges.Add(page, TRACKER_PAGE_SIZE);
		}
	}
}

void BufferCache::PublishReadHotCoalesced() {
	static const uint32_t interval = [] {
		const auto* const value = std::getenv("KYTY_PUBLISH_SLICES");
		if (value == nullptr || *value == '\0') {
			return 1u;
		}
		return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}();
	if (interval <= 1) {
		PublishReadHot();
		return;
	}
	if (++m_publish_slices < interval) {
		return;
	}
	m_publish_slices = 0;
	PublishReadHot();
}

void BufferCache::DecayReadHot() {
	m_hot_flip++;
	m_publish_slices = 0;
	const auto hot_flips = LabEnv("KYTY_HOT_FLIPS", DefaultHotFlips);
	std::erase_if(m_hot_pages, [this, hot_flips](const auto& page) {
		if (m_hot_flip - page.second <= hot_flips) {
			return false;
		}
		m_hot_ranges.Subtract(page.first, TRACKER_PAGE_SIZE);
		return true;
	});
}

void BufferCache::SupersedePublishes(uint64_t vaddr, uint64_t size) {
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto& batch: m_inflight_publishes) {
		for (auto& entry: *batch) {
			entry.superseded |=
			    Common::AlignDown(entry.vaddr, TRACKER_PAGE_SIZE) < last &&
			    first < Common::AlignUp(entry.vaddr + entry.size, TRACKER_PAGE_SIZE);
		}
	}
}

void BufferCache::SupersedeAllPublishes() {
	for (auto& batch: m_inflight_publishes) {
		for (auto& entry: *batch) {
			entry.superseded = true;
		}
	}
}

void BufferCache::PublishReadHot() {
	if (m_publish_ranges.Empty()) {
		return;
	}
	Common::PerfProbe::Scope publish_probe(Common::PerfProbe::Slot::PublishHot);
	struct Target {
		uint64_t vaddr;
		uint64_t offset;
		uint64_t size;
	};
	std::vector<std::pair<BufferId, std::vector<vk::BufferCopy>>> copies;
	std::vector<Target>                                           targets;
	auto     batch = std::make_shared<PublishBatch>();
	uint64_t total = 0;
	RangeSet retained;
	m_publish_ranges.ForEach([&](uint64_t start, uint64_t end) {
		const auto* owner = m_page_table.Find(start >> PageTable::kPageBits);
		if (owner == nullptr || IsBufferInvalid(*owner) ||
		    !m_slot_buffers[*owner].IsInBounds(start, end - start)) {
			return;
		}
		const auto id      = *owner;
		const auto address = m_slot_buffers[id].CpuAddress();
		m_gpu_modified_ranges.ForEachInRange(
		    start, end - start, [&](uint64_t first, uint64_t last) {
			    const auto bytes = last - first;
			    if (total + bytes > LabEnv("KYTY_PUBLISH_BUDGET", DefaultPublishBudget)) {
				    retained.Add(first, bytes);
				    return;
			    }
			    if (copies.empty() || copies.back().first != id) {
				    copies.emplace_back(id, std::vector<vk::BufferCopy> {});
			    }
			    copies.back().second.emplace_back(first - address, total, bytes);
			    targets.push_back({first, total, bytes});
			    batch->push_back({first, bytes, false});
			    total += Common::AlignUp(bytes, 64);
		    });
	});
	if (batch->empty()) {
		m_publish_ranges = std::move(retained);
		return;
	}
	auto [mapped, offset] = m_download_buffer.Map(total, 64);
	if (mapped == nullptr) {
		return;
	}
	m_publish_ranges = std::move(retained);
	m_download_buffer.Commit();
	auto& command = m_scheduler.Current();
	command.EndRendering();
	for (auto& [id, list]: copies) {
		for (auto& copy: list) {
			copy.dstOffset += offset;
		}
		RecordDownloadCopies(command.Handle(), m_slot_buffers[id], m_download_buffer, list, offset,
		                     total);
	}
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total, targets = std::move(targets)] {
		m_download_buffer.Invalidate(offset, total);
		for (const auto& target: targets) {
			Libs::LibKernel::Memory::WriteBacking(target.vaddr, mapped + target.offset,
			                                      target.size);
		}
	});
	m_inflight_publishes.push_back(batch);
	m_scheduler.DeferOperation([this, batch] { ConfirmPublish(batch); });
}

void BufferCache::ConfirmPublish(const std::shared_ptr<PublishBatch>& batch) {
	std::erase(m_inflight_publishes, batch);
	for (const auto& entry: *batch) {
		if (entry.superseded) {
			continue;
		}
		m_gpu_modified_ranges.Subtract(entry.vaddr, entry.size);
		ServeClearDirty(entry.vaddr, entry.size);
		ForgetColorMetadataChecks(entry.vaddr, entry.size);
	}
	for (const auto& entry: *batch) {
		if (entry.superseded) {
			continue;
		}
		const auto last = Common::AlignUp(entry.vaddr + entry.size, TRACKER_PAGE_SIZE);
		for (auto page = Common::AlignDown(entry.vaddr, TRACKER_PAGE_SIZE); page < last;
		     page += TRACKER_PAGE_SIZE) {
			if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
			    !AsyncReadbackIntersects(page, TRACKER_PAGE_SIZE) &&
			    m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
			}
		}
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Sync);
	// Fast path: most of the thousands of read-only obtains per frame have nothing to upload.
	// One lock-free tracker query is far cheaper than the create-on-miss region walk plus the
	// upload machinery. Written obtains must still run to mark the range GPU-dirty, and texel
	// buffers need the image synchronization check.
	if (!is_written && !is_texel_buffer && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		return false;
	}
	// ObtainBuffer reaches this path thousands of times per frame; a scratch vector per call
	// showed up as allocator traffic on the GPU thread.
	static thread_local std::vector<vk::BufferCopy> copies;
	copies.clear();
	uint64_t total_size = 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    const BufferUpload upload {&buffer, copies.size()};
		    UploadCopies({&upload, 1}, copies, total_size);
	    });
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::UploadCopies(std::span<const BufferUpload> uploads,
                               std::span<vk::BufferCopy> copies, uint64_t total_size) {
	if (copies.empty()) {
		return;
	}
	// After the early-out: the empty case is the common one (~2800 calls per frame).
	Common::PerfProbe::Scope probe(Common::PerfProbe::Slot::Upload);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    size_t first = 0;
		    for (const auto& upload: uploads) {
			    for (auto& copy: copies.subspan(first, upload.end - first)) {
				    const auto address = upload.buffer->CpuAddress() + copy.dstOffset;
				    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
					                                         copy.size)) {
					    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				    }
				    copy.srcOffset += base_offset;
			    }
			    first = upload.end;
		    }
	    });
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const vk::MemoryBarrier before {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                 vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eTransferWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before, 0, nullptr, 0, nullptr);
	size_t first = 0;
	for (const auto& upload: uploads) {
		native.copyBuffer(source.first->Handle(), upload.buffer->Handle(),
		                  static_cast<uint32_t>(upload.end - first), copies.data() + first);
		first = upload.end;
	}
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                               .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
	                                                vk::AccessFlagBits::eMemoryWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		if (Common::PerfProbe::LiteEnabled() && size <= 256 &&
		    (Common::PerfProbe::FrameIndexRef().load(std::memory_order_relaxed) % 50) == 0) {
			Common::PerfProbe::Trace("GW src=%d addr=0x%llx size=%llu was_dirty=%d tick=%llu",
			                         g_write_source, static_cast<unsigned long long>(vaddr),
			                         static_cast<unsigned long long>(size),
			                         m_gpu_modified_ranges.Intersects(vaddr, size) ? 1 : 0,
			                         static_cast<unsigned long long>(m_scheduler.CurrentTick()));
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		ServeMarkDirty(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		NoteRecentWrite(vaddr, size);
		NoteWriteTick(vaddr, size);
		SupersedePublishes(vaddr, size);
		m_hot_ranges.ForEachInRange(vaddr, size, [this](uint64_t start, uint64_t end) {
			m_publish_ranges.Add(start, end - start);
		});
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	g_write_source         = 1;
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	g_write_source         = 0;
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && m_texture_cache.MayContainImage(dst_vaddr, size) &&
	    !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) &&
	    (!m_texture_cache.MayContainImage(src_vaddr, size) ||
	     !m_texture_cache.FindImageFromRange(src_vaddr, size))) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	g_write_source = (src_memory && IsRegionGpuModified(src_vaddr, size)) ? 3 : 2;
	if (src_gds) {
		g_write_source = 4;
	}
	if (Common::PerfProbe::LiteEnabled() && size <= 256 &&
	    (Common::PerfProbe::FrameIndexRef().load(std::memory_order_relaxed) % 50) == 0) {
		Common::PerfProbe::Trace("GC dst=0x%llx src=0x%llx size=%llu dst_gpu=%d src_gpu=%d "
		                         "src_dirty=%d",
		                         static_cast<unsigned long long>(dst_vaddr),
		                         static_cast<unsigned long long>(src_vaddr),
		                         static_cast<unsigned long long>(size),
		                         dst_memory && IsRegionGpuModified(dst_vaddr, size) ? 1 : 0,
		                         src_memory && IsRegionGpuModified(src_vaddr, size) ? 1 : 0,
		                         src_memory && m_gpu_modified_ranges.Intersects(src_vaddr, size) ? 1
		                                                                                         : 0);
	}
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	g_write_source         = 0;
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	{
		Common::PerfProbe::CoarseScope wait(Common::PerfProbe::Slot::WaitDownload);
		Common::PerfProbe::Trace("GCDL buffers=%zu", dirty_buffers.size());
		m_scheduler.Wait(completion_tick);
	}
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	if (!is_write) {
		m_bda_sync.Add(begin, end - begin);
		return;
	}
	m_dma_accessed.ForEachInRange(begin, end - begin, [this](uint64_t first, uint64_t last) {
		m_bda_sync.Add(first, last - first);
	});
}

void BufferCache::MarkDmaAccess(const RangeSet& pages) {
	{
		std::lock_guard lock(m_bda_sync_mutex);
		pages.ForEach([this](uint64_t begin, uint64_t end) {
			m_dma_accessed.Add(begin, end - begin);
			m_bda_sync.Add(begin, end - begin);
		});
	}
	RangeSet                       unbacked;
	std::vector<vk::DeviceAddress> entries;
	pages.ForEach([&](uint64_t begin, uint64_t end) {
		entries.clear();
		for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
			const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
			if (owner == nullptr || !*owner) {
				LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
				unbacked.Add(address, CACHING_PAGESIZE);
				entries.push_back(0);
				continue;
			}
			const auto& buffer = m_slot_buffers[*owner];
			entries.push_back(buffer.BufferDeviceAddress() + buffer.Offset(address) + BDA_ACCESSED);
		}
		if (!unbacked.Contains(begin, end - begin)) {
			WriteDataBuffer(m_bda_pagetable_buffer, PageIndex(begin) * sizeof(vk::DeviceAddress),
			                entries.data(), entries.size() * sizeof(vk::DeviceAddress));
		}
	});
	unbacked.ForEach(
	    [this](uint64_t start, uint64_t end) { (void)FindBuffer(start, end - start); });
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	const auto queued = [this] {
		std::lock_guard lock(m_bda_sync_mutex);
		return std::exchange(m_bda_sync, {});
	}();
	std::vector<vk::BufferCopy> copies;
	std::vector<BufferUpload>   uploads;
	uint64_t                    total_size = 0;
	queued.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < finish; ++it) {
				auto&      buffer = m_slot_buffers[it->second];
				const auto first  = std::max(buffer.CpuAddress(), start);
				const auto last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
				if (first < last) {
					// Reads follow every range's protection; a write in between re-queues its page.
					m_memory_tracker.ForEachUploadRange(
					    first, last - first, false,
					    [&](uint64_t address, uint64_t bytes) noexcept {
						    if (uploads.empty() || uploads.back().buffer != &buffer) {
							    uploads.push_back({&buffer, 0});
						    }
						    copies.emplace_back(total_size, buffer.Offset(address), bytes);
						    uploads.back().end = copies.size();
						    total_size += bytes;
					    },
					    []() noexcept {});
				}
			}
		});
	});
	UploadCopies(uploads, copies, total_size);
}

} // namespace Libs::Graphics
