//  DRAFT — NOT FROZEN. Owned by the final review. Frozen at WP-1 (core/simd) /
//  WP-3 (Operator). Do not add fields, rename, or implement logic here.
//
//  DRAFT. Documents the contract WP-1 implements: every vectorized kernel has an
//  independently-written scalar twin returning identical results, e.g.
//    void add_f64_vec(const double* a, const double* b, double* out, size_t n);
//    void add_f64_scalar(const double* a, const double* b, double* out, size_t n);
//  No implementation here.
//
//  Notes for WP-1 (non-binding, orientation only):
//   * The vector path is the ONLY place Google Highway appears; the scalar twin
//     must be SEPARATE code (collapsing them makes scalar == vector prove
//     nothing — RIGOR.md rule 3).
//   * No vector width / cache size / core count / ISA may appear in a public
//     signature; Highway detects the target at runtime.
#pragma once

namespace qe::simd {

// Intentionally empty in WP-0: this header exists to carry the convention above
// to every kernel author. WP-1 adds the real vec/scalar twin declarations.

}  // namespace qe::simd
