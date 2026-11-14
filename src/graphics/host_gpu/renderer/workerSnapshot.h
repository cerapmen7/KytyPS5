#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WORKERSNAPSHOT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WORKERSNAPSHOT_H_

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "kernel/memory.h"

namespace Libs::Graphics {

// Per-draw copy of the guest bytes a preparation worker is allowed to read. The GPU thread
// captures the ranges before the command submission can reuse them. A worker that asks for
// anything outside these ranges fails preparation and falls back to the synchronous path;
// it never reads live guest memory.
class WorkerGuestSnapshot {
public:
	void Capture(uint64_t address, uint64_t size) {
		if (size == 0 || size > MaxRangeSize || m_ranges.size() >= MaxRanges) {
			return;
		}
		for (const auto& range: m_ranges) {
			if (address >= range.address && address + size <= range.address + range.bytes.size()) {
				return;
			}
		}
		Range range;
		range.address = address;
		range.bytes.resize(size);
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(address, range.bytes.data(), size)) {
			return;
		}
		m_ranges.push_back(std::move(range));
	}

	[[nodiscard]] bool Read(uint64_t address, std::span<uint32_t> values) const {
		const auto size = values.size_bytes();
		if (size == 0) {
			return true;
		}
		for (const auto& range: m_ranges) {
			if (address >= range.address && address + size <= range.address + range.bytes.size()) {
				std::memcpy(values.data(), range.bytes.data() + (address - range.address), size);
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] bool Empty() const noexcept { return m_ranges.empty(); }

private:
	static constexpr uint64_t MaxRangeSize = 1024;
	static constexpr size_t   MaxRanges    = 16;

	struct Range {
		uint64_t             address = 0;
		std::vector<uint8_t> bytes;
	};

	std::vector<Range> m_ranges;
};

// Set on a preparation worker around GetGraphicsPrograms. Null on the GPU thread, so the
// synchronous path keeps reading the live guest mapping.
inline thread_local const WorkerGuestSnapshot* g_worker_guest_snapshot = nullptr;

} // namespace Libs::Graphics

#endif
