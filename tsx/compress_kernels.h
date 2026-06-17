//  WP-14: the ONE genuinely-vectorizable step of the streaming decoder — the
//  zig-zag → signed transform — exposed as the twin pair the RIGOR.md rule-3 /
//  D17 convention mandates (simd/kernel_convention.h): a Highway `_vec` entry and
//  an independently-written `_scalar` twin with identical signatures.
//
//  WHY ONLY THIS STEP IS VECTORIZED (documented per the WP brief). The two integer
//  codecs (delta-of-delta for TS/I64, plain zig-zag for I32) decode in three
//  stages: (1) VARINT byte extraction — inherently sequential (each value's byte
//  length is data-dependent); (2) ZIG-ZAG decode — ELEMENT-WISE over the extracted
//  words, no cross-element dependence => the SIMD win, kernelized here; (3) the
//  delta-of-delta running double-prefix-sum — inherently sequential (value i needs
//  value i-1), kept scalar by the WP-5/WP-6 precedent. Gorilla XOR (F64) is
//  end-to-end sequential (variable-length bit blocks XOR'd against the prior value)
//  and has no vector form; it stays fully scalar. So stage (2) is the lone, honest
//  vector/scalar twin, and CompressedScan's DecodePath seam forces both paths on
//  identical compressed input for the scalar==vector check.
//
//  unzigzag: out[k] = (zz[k] >> 1) ^ (0 - (zz[k] & 1)).  Maps an unsigned zig-zag
//  word back to the two's-complement bit pattern of the signed value it encodes
//  (purely unsigned bit ops — no signed shifts, UBSan-clean). `out` MUST NOT alias
//  `zz` only if the caller needs `zz` afterwards; the transform itself is
//  position-local so in-place is safe, but the kernels make no aliasing promise.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::tsx {

// Vector path (Highway dynamic dispatch lives behind this; see compress_kernels.cpp).
void unzigzag_vec(const std::uint64_t* zz, std::size_t n, std::uint64_t* out);
// Independently-written scalar twin (plain C++; see compress_scalar.cpp).
void unzigzag_scalar(const std::uint64_t* zz, std::size_t n, std::uint64_t* out);

}  // namespace qe::tsx
