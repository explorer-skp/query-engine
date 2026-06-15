//  WP-9: the THREE new planted mutants the hardened mutation catalog adds on top
//  of the per-WP mutants already shipped (simd/expr/ops/plan *_mutants.*). Each is
//  a deliberately-broken twin of a real engine step, drawn from a §12 hotspot the
//  earlier WPs did not already plant a mutant for, plus the WP-2 carry-forward.
//  "A checker that cannot fail proves nothing" — these exist so the catalog's
//  meta-test can SHOW the differential / self-test suite FLAGS each one.
//
//  They are TEST-ONLY (compiled into the catalog/meta-test target, NEVER linked
//  into any engine library) and are pure, side-effect-free helpers so the catalog
//  check() can run them against the real engine result and the DuckDB/reference
//  oracle.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::catalog::mutant {

// CARRY-FORWARD (WP-2 pinned bug, §12 float/cast): F64 -> I64 cast that rounds by
// adding 0.5 (copysign for negatives) before truncating. This CORRUPTS integral
// doubles with |x| >= 2^52: at that magnitude the ULP is >= 1, so `x + 0.5` rounds
// to a DIFFERENT integer than x (round-half-to-even pushes odd integers up by one).
// The real engine cast (expr/cast_kernels.cpp::RoundAway) was fixed to trunc + an
// exact-fractional bump; this mutant is the pre-fix behavior. Caught by the
// project-CAST differential vs DuckDB (which casts the integral double exactly).
void cast_f64_to_i64_plus_half(const double* in, std::int64_t* out,
                               std::size_t n);

// §12 aggregation integer overflow: a global SUM that accumulates into a 32-bit
// register instead of the engine's I64 accumulator. Wraps (mod 2^32, returned
// sign-extended from int32) once the running total exceeds the int32 range, so the
// result diverges from DuckDB's exact BIGINT SUM. The wraparound is done through
// unsigned arithmetic so the MUTANT itself is UB-free (UBSan stays green) — the
// bug is a WRONG VALUE, not undefined behavior.
std::int64_t sum_i64_wrapping_i32(const std::int64_t* in, std::size_t n);

// §12 selection-vector index aliasing after compaction: an IN-PLACE gather
// (out aliases in) — `buf[k] = buf[idx[k]]` over the same buffer. This is exactly
// what the real compact_column refuses to do (it is out-of-place by construction,
// core/owned_batch.h): for any non-monotone permutation a slot is overwritten
// before a later index reads it, so the result is corrupted. Caught by diffing
// against the real (out-of-place) compact_column on a reversing permutation.
void gather_in_place_aliased(std::int64_t* buf, const std::uint32_t* idx,
                             std::size_t n);

}  // namespace qe::catalog::mutant
