#ifndef KYTY_COMMON_PERF_PROBE_H_
#define KYTY_COMMON_PERF_PROBE_H_

// Lightweight in-process phase profiler for the graphics hot path.
//
// KYTY_PROBE=0 or unset : disabled (zero work beyond one branch in Scope).
// KYTY_PROBE=1          : per-frame CSV in KYTY_PROBE_DIR (default ".").
// KYTY_PROBE=2          : level 1 + slowframes.csv with the top leaf phases.
// KYTY_PROBE=3          : reserved for per-draw detail.
//
// The per-frame CSV is written from the present thread by EndFrame(); the
// phase counters themselves are cumulative relaxed atomics updated by the GPU
// thread. Tick() keeps printing the old one-line-per-second summary for live
// feedback, but no longer resets the counters.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

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
	OpDma,
	OpAcquire,
	OpCopy,
	OpIndirect,
	OpWait,
	OpSem,
	OpNop,
	StaticZero,
	StaticApply,
	StaticDetect,
	PsStaticHit,
	PsStaticMiss,
	MatRes,
	MatWalk,
	MatBuf,
	MatImg,
	MatSpec,
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
	FindImage,
	MatClear,
	Readback,
	Sync,
	Upload,
	Present,
	GpuWait,
	GuestFrame,
	GuestWait,
	DrawPrefix,
	Dispatch,
	DispatchProgram,
	DispatchBindings,
	DispatchCommit,
	DispatchBarrier,
	GraphicsPrograms,
	DescWrite,
	DescAlloc,
	PublishHot,
	WaitFinish,
	WaitDownload,
	WaitFault,
	WaitOccl,
	StaticDecode,
	Count,
};

inline constexpr uint32_t SlotCount = static_cast<uint32_t>(Slot::Count);

inline const char* Name(Slot slot) noexcept {
	static constexpr const char* names[] = {
	    "draw_total",  "programs",     "resolve",      "bindings",     "vertexbuf",
	    "pipeline",    "rtargets",     "commit",       "dynstate",     "drawcall",
	    "reset",       "flush",        "submit",       "frame_busy",   "drawindex",
	    "pending",     "vsprep",       "psprep",       "vsmod",        "psmod",
	    "stages",      "findbuf",      "rebindimg",    "rebindbuf",    "maplookup",
	    "staticinfo",  "gc",           "pm4",          "flipwait",     "process",
	    "psstatic",    "psparams",     "pm4handler",   "relmem",       "wrdata",
	    "evt",         "op_sh",        "op_uc",        "op_cfg",       "op_draw",
	    "op_other",    "op_dma",       "op_acquire",   "op_copy",      "op_indirect",
	    "op_wait",     "op_sem",       "op_nop",       "staticzero",   "staticapply",
	    "staticdetect",
	    "psstatic_hit","psstatic_miss","matres",       "matwalk",      "matbuf",
	    "matimg",      "matspec",      "natbuf",       "natup",
	    "natobtain",   "gpfind",       "gpkey",        "hash_time",    "hash_hit",
	    "hash_miss",   "statickey",    "staticcopy",   "depthonly",    "layered",
	    "mrt",         "findimage",    "matclear",     "readback",     "sync",
	    "upload",      "present",      "gpuwait",      "guestframe",   "guestwait",
	    "drawprefix",  "dispatch",     "dispprog",     "dispbind",     "dispcommit",
	    "dispbar",     "gpprep",       "descwrite",    "descalloc",    "publishhot",
	    "waitfinish",
	    "waitdownload", "waitfault",   "waitoccl",     "staticdecode",
	};
	static_assert(sizeof(names) / sizeof(names[0]) == SlotCount);
	return names[static_cast<uint32_t>(slot)];
}

inline int LevelRef() noexcept {
	static const int level = [] {
		const char* env = std::getenv("KYTY_PROBE");
		if (env == nullptr || *env == '\0') {
			return 0;
		}
		return std::atoi(env);
	}();
	return level;
}

inline bool Enabled() noexcept {
	return LevelRef() > 0;
}

inline int Level() noexcept {
	return LevelRef();
}

inline uint64_t Now() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count());
}

inline uint64_t WallMs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(
	        std::chrono::system_clock::now().time_since_epoch())
	        .count());
}

inline std::atomic<uint64_t>& NsRef(Slot slot) noexcept {
	static std::atomic<uint64_t> values[SlotCount] {};
	return values[static_cast<uint32_t>(slot)];
}

inline std::atomic<uint64_t>& CallsRef(Slot slot) noexcept {
	static std::atomic<uint64_t> values[SlotCount] {};
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

// Same as Scope, but the measured range can be closed before the object dies so a
// long function can time only its prefix.
class Span final {
public:
	explicit Span(Slot slot) noexcept : m_slot(slot), m_start(Enabled() ? Now() : 0) {}
	~Span() { Stop(); }
	void Stop() noexcept {
		if (m_start != 0) {
			Add(m_slot, Now() - m_start);
			m_start = 0;
		}
	}
	Span(const Span&)            = delete;
	Span& operator=(const Span&) = delete;

private:
	Slot     m_slot;
	uint64_t m_start;
};

struct Snapshot {
	uint64_t ns[SlotCount] {};
	uint64_t calls[SlotCount] {};
};

inline Snapshot Capture() noexcept {
	Snapshot snapshot;
	for (uint32_t i = 0; i < SlotCount; i++) {
		snapshot.ns[i]    = NsRef(static_cast<Slot>(i)).load(std::memory_order_relaxed);
		snapshot.calls[i] = CallsRef(static_cast<Slot>(i)).load(std::memory_order_relaxed);
	}
	return snapshot;
}

inline Snapshot Delta(const Snapshot& now, const Snapshot& prev) noexcept {
	Snapshot delta;
	for (uint32_t i = 0; i < SlotCount; i++) {
		delta.ns[i]    = now.ns[i] - prev.ns[i];
		delta.calls[i] = now.calls[i] - prev.calls[i];
	}
	return delta;
}

// Top-level phases that contain other slots. Excluded from "top leaf" picks.
inline bool IsAggregate(Slot slot) noexcept {
	switch (slot) {
	case Slot::DrawTotal:
	case Slot::DrawIndexTotal:
	case Slot::Process:
	case Slot::Pm4:
	case Slot::Programs:
	case Slot::Bindings:
	case Slot::StaticInfo:
	case Slot::VertexBuffers:
	case Slot::Flush:
	case Slot::Dispatch:
	case Slot::GraphicsPrograms:
		return true;
	default:
		return false;
	}
}

inline Slot TopLeaf(const Snapshot& delta, double& out_ms) noexcept {
	Slot   best    = Slot::Count;
	double best_ms = 0.0;
	for (uint32_t i = 0; i < SlotCount; i++) {
		const auto slot = static_cast<Slot>(i);
		if (IsAggregate(slot)) {
			continue;
		}
		const auto ms = static_cast<double>(delta.ns[i]) / 1e6;
		if (ms > best_ms) {
			best_ms = ms;
			best    = slot;
		}
	}
	if (best == Slot::Count) {
		best    = Slot::DrawTotal;
		best_ms = static_cast<double>(delta.ns[static_cast<uint32_t>(Slot::DrawTotal)]) / 1e6;
	}
	out_ms = best_ms;
	return best;
}

// Called from the graphics thread after each draw. Prints at most once per
// second, using deltas against the previous print; the atomics stay cumulative.
inline void Tick() noexcept {
	if (!Enabled()) {
		return;
	}
	static uint32_t counter = 0;
	if ((++counter & 31u) != 0) {
		return;
	}
	static uint64_t last_print = 0;
	static Snapshot prev {};
	const auto      now        = Now();
	if (last_print == 0) {
		last_print = now;
		prev       = Capture();
		return;
	}
	if (now - last_print < 1'000'000'000ull) {
		return;
	}
	const auto elapsed_s = static_cast<double>(now - last_print) / 1e9;
	last_print           = now;
	const auto cur       = Capture();
	const auto delta     = Delta(cur, prev);
	prev                 = cur;
	if (delta.calls[static_cast<uint32_t>(Slot::DrawTotal)] == 0) {
		return;
	}
	std::printf("PROBE window=%.2fs", elapsed_s);
	for (uint32_t i = 0; i < SlotCount; i++) {
		if (delta.calls[i] == 0) {
			continue;
		}
		std::printf(" %s=%.1fms/%llu", Name(static_cast<Slot>(i)),
		            static_cast<double>(delta.ns[i]) / 1e6,
		            static_cast<unsigned long long>(delta.calls[i]));
	}
	std::printf("\n");
	std::fflush(stdout);
}

struct CsvFile final {
	std::FILE* file = nullptr;
	~CsvFile() {
		if (file != nullptr) {
			std::fclose(file);
			file = nullptr;
		}
	}
};

inline CsvFile& FrameCsv() noexcept {
	static CsvFile csv;
	return csv;
}

inline CsvFile& SlowCsv() noexcept {
	static CsvFile csv;
	return csv;
}

inline std::string ProbeDir() noexcept {
	const char* env = std::getenv("KYTY_PROBE_DIR");
	return (env != nullptr && *env != '\0') ? std::string(env) : std::string(".");
}

inline void WriteFrameHeader(std::FILE* file) noexcept {
	std::fprintf(file, "frame,wall_ms,mono_ns,frame_ms,fps,draws,slow,top_slot,top_ms");
	for (uint32_t i = 0; i < SlotCount; i++) {
		std::fprintf(file, ",%s_ms,%s_calls", Name(static_cast<Slot>(i)),
		             Name(static_cast<Slot>(i)));
	}
	std::fprintf(file, "\n");
}

inline void WriteSlowHeader(std::FILE* file) noexcept {
	std::fprintf(file, "frame,wall_ms,frame_ms,fps,draws");
	for (uint32_t i = 0; i < 5; i++) {
		std::fprintf(file, ",top%u_slot,top%u_ms", i + 1, i + 1);
	}
	std::fprintf(file, "\n");
}

inline void OpenCsv() noexcept {
	if (FrameCsv().file != nullptr) {
		return;
	}
	std::error_code ec;
	const auto      dir = ProbeDir();
	std::filesystem::create_directories(dir, ec);
	const auto path = std::filesystem::path(dir) / "frames.csv";
	FrameCsv().file = std::fopen(path.string().c_str(), "w");
	if (FrameCsv().file != nullptr) {
		WriteFrameHeader(FrameCsv().file);
	} else {
		std::fprintf(stderr, "perfProbe: cannot open %s\n", path.string().c_str());
	}
	if (Level() >= 2) {
		const auto slow_path = std::filesystem::path(dir) / "slowframes.csv";
		SlowCsv().file       = std::fopen(slow_path.string().c_str(), "w");
		if (SlowCsv().file != nullptr) {
			WriteSlowHeader(SlowCsv().file);
		}
	}
}

// Called from the present thread after a guest frame is presented.
inline void EndFrame() noexcept {
	if (Level() < 1) {
		return;
	}
	static Snapshot prev {};
	static uint64_t last_ns       = 0;
	static uint64_t frame_index   = 0;
	static uint64_t flush_counter = 0;
	static double   ema           = 0.0;
	static bool     ema_ready     = false;

	const auto now  = Now();
	const auto wall = WallMs();
	const auto cur  = Capture();
	const auto d    = Delta(cur, prev);
	prev            = cur;

	const double frame_ms = (last_ns == 0) ? 0.0 : static_cast<double>(now - last_ns) / 1e6;
	last_ns               = now;

	if (!ema_ready) {
		ema       = frame_ms;
		ema_ready = true;
	} else {
		ema = 0.95 * ema + 0.05 * frame_ms;
	}
	const bool   slow    = frame_index > 30 && frame_ms > 1.5 * ema;
	const auto   draws   = d.calls[static_cast<uint32_t>(Slot::DrawTotal)];
	double       top_ms  = 0.0;
	const auto   top     = TopLeaf(d, top_ms);
	const double fps     = frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0;

	OpenCsv();
	if (FrameCsv().file != nullptr) {
		std::fprintf(FrameCsv().file, "%llu,%llu,%llu,%.3f,%.2f,%llu,%d,%s,%.3f",
		             static_cast<unsigned long long>(frame_index),
		             static_cast<unsigned long long>(wall),
		             static_cast<unsigned long long>(now), frame_ms, fps,
		             static_cast<unsigned long long>(draws), slow ? 1 : 0, Name(top),
		             top_ms);
		for (uint32_t i = 0; i < SlotCount; i++) {
			std::fprintf(FrameCsv().file, ",%.3f,%llu",
			             static_cast<double>(d.ns[i]) / 1e6,
			             static_cast<unsigned long long>(d.calls[i]));
		}
		std::fprintf(FrameCsv().file, "\n");
		if (++flush_counter >= 30) {
			std::fflush(FrameCsv().file);
			flush_counter = 0;
		}
	}

	if (slow && SlowCsv().file != nullptr) {
		// Five slowest leaf phases for this frame.
		Slot   best[5] = {Slot::Count, Slot::Count, Slot::Count, Slot::Count, Slot::Count};
		double best_ms[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
		for (uint32_t i = 0; i < SlotCount; i++) {
			const auto slot = static_cast<Slot>(i);
			if (IsAggregate(slot)) {
				continue;
			}
			const double ms = static_cast<double>(d.ns[i]) / 1e6;
			for (uint32_t k = 0; k < 5; k++) {
				if (ms > best_ms[k]) {
					for (uint32_t j = 4; j > k; j--) {
						best_ms[j] = best_ms[j - 1];
						best[j]    = best[j - 1];
					}
					best_ms[k] = ms;
					best[k]    = slot;
					break;
				}
			}
		}
		std::fprintf(SlowCsv().file, "%llu,%llu,%.3f,%.2f,%llu",
		             static_cast<unsigned long long>(frame_index),
		             static_cast<unsigned long long>(wall), frame_ms, fps,
		             static_cast<unsigned long long>(draws));
		for (uint32_t k = 0; k < 5; k++) {
			std::fprintf(SlowCsv().file, ",%s,%.3f",
			             best[k] == Slot::Count ? "none" : Name(best[k]), best_ms[k]);
		}
		std::fprintf(SlowCsv().file, "\n");
		std::fflush(SlowCsv().file);
	}

	frame_index++;
}

} // namespace Common::PerfProbe

#endif /* KYTY_COMMON_PERF_PROBE_H_ */
