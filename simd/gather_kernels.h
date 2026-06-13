//  WP-1: selection-vector GATHER kernels — the vectorized technique at the heart
//  of selection-vector flow (decision D6), in the same twin form as the validity
//  kernels (simd/kernel_convention.h).
//
//  out[k] = src[idx[k]] for k in [0, n). `idx == nullptr` means the dense
//  identity (out[k] = src[k]) — a plain copy. Gathers are typed by lane WIDTH,
//  not by semantic Type: I32 reinterprets to gather32, while I64/F64/TS
//  reinterpret to gather64 (BOOL is one byte — use gather8_scalar). Higher layers
//  do the reinterpret (see core/owned_batch.h compact_column).
//
//  ALIASING (the §12 hotspot): `out` MUST NOT alias `src`. Compaction under a
//  selection vector is OUT-OF-PLACE; in-place compaction is undefined because a
//  gather may read src[idx[k]] for idx[k] > k after that slot was overwritten.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::simd {

void gather32_vec(const std::uint32_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint32_t* out);
void gather32_scalar(const std::uint32_t* src, const std::uint32_t* idx,
                     std::size_t n, std::uint32_t* out);

void gather64_vec(const std::uint64_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint64_t* out);
void gather64_scalar(const std::uint64_t* src, const std::uint32_t* idx,
                     std::size_t n, std::uint64_t* out);

// One byte per element (BOOL). Scalar only — no vector twin — because a 1-byte
// lane gather has no meaningful SIMD form here; documented, not an omission.
void gather8_scalar(const std::uint8_t* src, const std::uint32_t* idx,
                    std::size_t n, std::uint8_t* out);

}  // namespace qe::simd
