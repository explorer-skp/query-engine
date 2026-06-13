//  Frozen at WP-1 (core/simd). The kernel-authoring convention every later WP
//  follows, plus the one runtime-detection accessor the format layer needs.
//
//  THE TWIN CONVENTION (RIGOR.md rule 3 / decision D17), binding on every WP:
//
//   * Every vectorized kernel ships with an independently-written SCALAR TWIN
//     that returns identical results. The two are SEPARATE code paths in
//     SEPARATE translation units (vector path: *_kernels.cpp built with Highway;
//     scalar path: *_scalar.cpp, plain C++). Collapsing them — e.g. one
//     templated body selected by a flag — makes `scalar == vector` prove
//     nothing, so it is forbidden.
//
//   * Naming: a kernel `foo` exposes exactly two free functions with identical
//     signatures, `qe::simd::foo_vec(...)` and `qe::simd::foo_scalar(...)`. The
//     `_vec` entry is a normal C++ function that internally performs Highway
//     dynamic dispatch (HWY_DYNAMIC_DISPATCH); Highway appears ONLY behind it.
//
//   * No vector width / cache size / ISA / core count may appear in a public
//     signature or be hardcoded in a body. Width is whatever Highway selects at
//     runtime; any alignment a kernel needs derives from runtime_vector_bytes()
//     (below) or simd::required_alignment_bytes() (simd/aligned_alloc.h) — never
//     a literal.
//
//   Example (WP-2's arithmetic kernels will look exactly like this):
//     void add_f64_vec(const double* a, const double* b, double* out, size_t n);
//     void add_f64_scalar(const double* a, const double* b, double* out, size_t n);
//
//  WP-1's concrete kernels following this convention: simd/validity_kernels.h
//  (validity-bitmap ops) and simd/gather_kernels.h (selection-vector gather).
#pragma once

#include <cstddef>

namespace qe::simd {

// The width, in bytes, of the widest SIMD vector Google Highway dispatches to on
// THIS machine at runtime (e.g. 16 on NEON, 64 on AVX-512). Detected via
// Highway — never hardcoded. This is the value Buffer alignment is derived from
// (see simd/aligned_alloc.h), so the x86 box needs zero code changes.
std::size_t runtime_vector_bytes();

}  // namespace qe::simd
