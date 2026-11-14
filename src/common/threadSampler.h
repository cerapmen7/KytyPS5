#ifndef KYTY_COMMON_THREAD_SAMPLER_H_
#define KYTY_COMMON_THREAD_SAMPLER_H_

// Opt-in statistical profiler for the emulator's hot threads.
//
// KYTY_SAMPLE=1 starts a 1 kHz sampler that suspends each registered thread,
// reads its instruction pointer and resumes it. Counts are aggregated per
// thread and per module-relative address, then resolved with DbgHelp at exit
// and written to <KYTY_PROBE_DIR>/sampler.csv.
//
// The sampler is a diagnostic tool: it costs a few microseconds per sample and
// is disabled unless KYTY_SAMPLE is set. It is header-only on purpose: the
// emulator's CMake globs do not include src/common.

#include <windows.h>

#include <dbghelp.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "winmm.lib")

namespace Common::Sampler {

struct ThreadEntry {
	std::string name;
	DWORD       tid    = 0;
	HANDLE      handle = nullptr;
};

struct State {
	std::mutex                mutex;
	std::vector<ThreadEntry>  threads;
	std::mutex                counts_mutex;
	std::map<std::pair<DWORD, uint64_t>, uint64_t> counts;
	std::atomic<bool>         running {false};
	std::thread               worker;
	uint64_t                  base   = 0;
	bool                      dumped = false;
};

inline State& Get() noexcept {
	static State* state = new State();
	return *state;
}

inline void Start() noexcept;

inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SAMPLE");
		return value != nullptr && *value != '\0' && std::strtoul(value, nullptr, 10) != 0;
	}();
	return enabled;
}

inline std::string OutputDir() noexcept {
	const char* dir = std::getenv("KYTY_PROBE_DIR");
	return (dir != nullptr && *dir != '\0') ? std::string(dir) : std::string(".");
}

// Registers the calling thread under a readable name. Called once per thread.
inline void RegisterThread(const char* name) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&     state = Get();
	HANDLE    dup   = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &dup, 0,
	                    FALSE, DUPLICATE_SAME_ACCESS) == 0) {
		return;
	}
	{
		std::lock_guard lock(state.mutex);
		state.threads.push_back({name, GetCurrentThreadId(), dup});
	}
	Start();
}

inline void Dump() noexcept {
	auto& state = Get();
	std::map<std::pair<DWORD, uint64_t>, uint64_t> counts;
	{
		std::lock_guard lock(state.counts_mutex);
		counts = state.counts;
	}
	if (counts.empty()) {
		return;
	}

	// Aggregate per (thread, address), then resolve.
	struct Row {
		std::string thread;
		uint64_t    address = 0;
		uint64_t    count   = 0;
		std::string symbol;
	};
	std::map<DWORD, std::string> names;
	{
		std::lock_guard lock(state.mutex);
		for (const auto& entry: state.threads) {
			names[entry.tid] = entry.name;
		}
	}

	std::vector<Row> rows;
	rows.reserve(counts.size());
	for (const auto& [key, count]: counts) {
		Row row;
		row.thread  = names[key.first];
		row.address = key.second;
		row.count   = count;
		rows.push_back(std::move(row));
	}

	HANDLE process = GetCurrentProcess();
	SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
	const bool symbols = SymInitialize(process, nullptr, TRUE) != 0;

	for (auto& row: rows) {
		if (!symbols) {
			row.symbol = "?";
			continue;
		}
		alignas(SYMBOL_INFO) char        storage[sizeof(SYMBOL_INFO) + 512] {};
		auto*                            info = reinterpret_cast<SYMBOL_INFO*>(storage);
		info->SizeOfStruct                   = sizeof(SYMBOL_INFO);
		info->MaxNameLen                     = 511;
		DWORD64 displacement                 = 0;
		const DWORD64 address = row.address;
		if (SymFromAddr(process, address, &displacement, info) != 0) {
			row.symbol = std::string(info->Name) + "+0x" +
			             [&] {
				             char buffer[32] {};
				             std::snprintf(buffer, sizeof(buffer), "%llx",
				                           static_cast<unsigned long long>(displacement));
				             return std::string(buffer);
			             }();
		} else {
			char buffer[64] {};
			std::snprintf(buffer, sizeof(buffer), "0x%llx",
			              static_cast<unsigned long long>(row.address));
			row.symbol = buffer;
		}
	}
	if (symbols) {
		SymCleanup(process);
	}

	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
		if (a.thread != b.thread) return a.thread < b.thread;
		return a.count > b.count;
	});

	const auto path = OutputDir() + "/sampler.csv";
	std::FILE* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "thread,address,samples,symbol\n");
	for (const auto& row: rows) {
		std::fprintf(file, "%s,0x%llx,%llu,%s\n", row.thread.c_str(),
		             static_cast<unsigned long long>(row.address),
		             static_cast<unsigned long long>(row.count), row.symbol.c_str());
	}
	std::fclose(file);
}

inline void Start() noexcept {
	if (!Enabled()) {
		return;
	}
	auto& state = Get();
	if (state.running.exchange(true)) {
		return;
	}
	std::printf("SAMPLER: enabled, writing %s/sampler.csv\n", OutputDir().c_str());
	std::fflush(stdout);
	state.base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	std::atexit([] { Dump(); });
	state.worker = std::thread([] {
		timeBeginPeriod(1);
		auto& s = Get();
		// Sampling suspends threads, so doing it during boot turns lock convoys into a crawl.
		// A delay lets a diagnostic session start in gameplay instead.
		const auto delay_sec = [] {
			const char* value = std::getenv("KYTY_SAMPLE_DELAY_SEC");
			return (value != nullptr && *value != '\0')
			           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
			           : 0u;
		}();
		for (uint32_t waited = 0; waited < delay_sec && s.running.load(std::memory_order_relaxed);
		     waited++) {
			Sleep(1000);
		}
		if (!s.running.load(std::memory_order_relaxed)) {
			timeEndPeriod(1);
			return;
		}
		const auto interval_ms = [] {
			const char* value = std::getenv("KYTY_SAMPLE_MS");
			return (value != nullptr && *value != '\0')
			           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
			           : 1u;
		}();
		uint32_t iterations = 0;
		while (s.running.load(std::memory_order_relaxed)) {
			// Collect the instruction pointers first and only touch the heap after every
			// sampled thread has been resumed: a suspended thread may hold the process heap
			// lock, and allocating under SuspendThread would deadlock the process.
			uint64_t keys[16] {};
			DWORD    tids[16] {};
			uint32_t count = 0;
			{
				std::lock_guard lock(s.mutex);
				for (const auto& entry: s.threads) {
					if (count >= std::size(keys)) {
						break;
					}
					if (SuspendThread(entry.handle) == static_cast<DWORD>(-1)) {
						continue;
					}
					CONTEXT context {};
					context.ContextFlags = CONTEXT_CONTROL;
					if (GetThreadContext(entry.handle, &context) != 0) {
						keys[count] = static_cast<uint64_t>(context.Rip);
						tids[count] = entry.tid;
						count++;
					}
					ResumeThread(entry.handle);
				}
			}
			if (count != 0) {
				std::lock_guard counts_lock(s.counts_mutex);
				for (uint32_t i = 0; i < count; i++) {
					s.counts[{tids[i], keys[i]}]++;
				}
			}
			if (++iterations % 5000 == 0) {
				Dump();
			}
			Sleep(interval_ms);
		}
		timeEndPeriod(1);
	});
}

inline void Stop() noexcept {
	auto& state = Get();
	if (!state.running.exchange(false)) {
		return;
	}
	if (state.worker.joinable()) {
		state.worker.join();
	}
	Dump();
}

} // namespace Common::Sampler

#endif /* KYTY_COMMON_THREAD_SAMPLER_H_ */
