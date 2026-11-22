#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX as a zero read and resumes after it.
void SkipLoad(void* native_context, size_t length);

// Reads `size` guest bytes at `address` into `out` when they may be served without the normal
// fault path; returns false to fall back to it.
using ServeReadFunc = bool (*)(uint64_t address, uint64_t size, void* out);
// Completes a faulting read whose only memory operand is a plain source (MOV family, SSE/AVX
// 128-bit loads, CMP, TEST) with bytes from `serve`, then resumes after the instruction.
// Returns false, changing nothing, for any other instruction or when `serve` declines.
[[nodiscard]] bool TryEmulateServedLoad(void* native_context, uint64_t fault_address,
                                        ServeReadFunc serve, uint32_t* mnemonic_out);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
