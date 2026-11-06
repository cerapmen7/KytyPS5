#ifndef KYTY_COMMON_PERF_PROBE_H_
#define KYTY_COMMON_PERF_PROBE_H_

// Lightweight in-process phase profiler for the graphics hot path.
// Enabled by default in release builds; set KYTY_PROBE=0 to disable.
// Prints a one-line summary at most once per second while draws are running.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Common::PerfProbe {

enum class Slot : uint32_t {
	DrawTotal,
	Programs,
	ResolveTargets,
	Bindings,
	VertexBuffers,
	Pipeline,
	RenderTargets,
	CommitBindings,
	DynamicState,
	DrawCall,
	ResetBindings,
	Flush,
	Submit,
	FrameBusy,
	DrawIndexTotal,
	PendingOps,
	VsPrepare,
	PsPrepare,
	VsModule,
	PsModule,
	PrepareStages,
	FindBuffers,
	RebindImages,
	RebindBuffers,
	MapLookup,
	StaticInfo,
	Gc,
	Pm4,
	FlipWait,
	Process,
	PsStatic,
	PsParams,
	Pm4Handler,
	RelMem,
	WrData,
	Evt,
	OpSh,
	OpUc,
	OpCfg,
	OpDraw,
	OpOther,
	OpNop,
	StaticZero,
	StaticApply,
	StaticDetect,
	PsStaticHit,
	PsStaticMiss,
	MatRes,
	NatBuf,
	NatUp,
	NatObtain,
	GpFind,
	GpKey,
	HashTime,
	HashHit,
	HashMiss,
	StaticKey,
	StaticCopy,
	DepthOnlyDraw,
	LayeredDraw,
	MrtDraw,
	Count,
};

inline const char* Name(Slot slot) noexcept {
	static constexpr const char* names[] = {
	    "draw_total", "programs",   "resolve",   "bindings",     "vertexbuf",
	    "pipeline",   "rtargets",   "commit",    "dynstate",     "drawcall",
	    "reset",      "flush",      "submit",    "frame_busy",   "drawindex",
	    "pending",    "vsprep",     "psprep",    "vsmod",        "psmod",
	    "stages",     "findbuf",    "rebindimg", "rebindbuf",    "maplookup",
	    "staticinfo", "gc",         "pm4",       "flipwait",     "process",
	    "psstatic",   "psparams",   "pm4handler", "relmem",      "wrdata",
	    "evt",        "op_sh",        "op_uc",         "op_cfg",       "op_draw",
	    "op_other",   "op_nop",       "staticzero",    "staticapply",  "staticdetect",
	    "psstatic_hit", "psstatic_miss", "matres",     "natbuf",       "natup",
	    "natobtain",  "gpfind",      "gpkey",        "hash_time",    "hash_hit",
	    "hash_miss",  "statickey",   "staticcopy",   "depthonly",    "layered",
	    "mrt",
	};
	return names[static_cast<uint32_t>(slot)];
}

inline bool& FlagRef() noexcept {
	static bool flag = [] {
		const char* env = std::getenv("KYTY_PROBE");
		return env == nullptr || env[0] != '0';
	}();
	return flag;
}

inline bool Enabled() noexcept {
	return FlagRef();
}

inline uint64_t Now() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count());
}

inline std::atomic<uint64_t>& NsRef(Slot slot) noexcept {
	static std::atomic<uint64_t> values[static_cast<uint32_t>(Slot::Count)] {};
	return values[static_cast<uint32_t>(slot)];
}

inline std::atomic<uint64_t>& CallsRef(Slot slot) noexcept {
	static std::atomic<uint64_t> values[static_cast<uint32_t>(Slot::Count)] {};
	return values[static_cast<uint32_t>(slot)];
}

inline void Add(Slot slot, uint64_t ns) noexcept {
	NsRef(slot).fetch_add(ns, std::memory_order_relaxed);
	CallsRef(slot).fetch_add(1, std::memory_order_relaxed);
}

// Accumulates into one slot; no per-call atomics on the read path.
class Scope final {
public:
	explicit Scope(Slot slot) noexcept : m_slot(slot), m_start(Enabled() ? Now() : 0) {}
	~Scope() {
		if (m_start != 0) {
			Add(m_slot, Now() - m_start);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Slot     m_slot;
	uint64_t m_start;
};

// Called from the graphics thread after each draw. Prints at most once per second.
inline void Tick() noexcept {
	if (!Enabled()) {
		return;
	}
	static uint64_t last_print = 0;
	static uint64_t window_start = 0;
	const auto      now        = Now();
	if (window_start == 0) {
		window_start = now;
		last_print   = now;
		return;
	}
	if (now - last_print < 1'000'000'000ull) {
		return;
	}
	const auto elapsed_s = static_cast<double>(now - last_print) / 1e9;
	last_print           = now;
	const auto draws     = CallsRef(Slot::DrawTotal).load(std::memory_order_relaxed);
	if (draws == 0) {
		return;
	}
	std::printf("PROBE window=%.2fs", elapsed_s);
	for (uint32_t i = 0; i < static_cast<uint32_t>(Slot::Count); i++) {
		const auto slot  = static_cast<Slot>(i);
		const auto ns    = NsRef(slot).load(std::memory_order_relaxed);
		const auto calls = CallsRef(slot).load(std::memory_order_relaxed);
		if (calls == 0) {
			continue;
		}
		std::printf(" %s=%.1fms/%llu", Name(slot), static_cast<double>(ns) / 1e6,
		            static_cast<unsigned long long>(calls));
	}
	std::printf("\n");
	std::fflush(stdout);
	for (uint32_t i = 0; i < static_cast<uint32_t>(Slot::Count); i++) {
		NsRef(static_cast<Slot>(i)).store(0, std::memory_order_relaxed);
		CallsRef(static_cast<Slot>(i)).store(0, std::memory_order_relaxed);
	}
}

} // namespace Common::PerfProbe

#endif /* KYTY_COMMON_PERF_PROBE_H_ */
