//  WP-5: INTERNAL kernel seam for hash aggregation — NOT a frozen contract (the
//  analog of expr/kernels.h / ops/hash_kernels.h). These are the genuinely
//  data-parallel, SIMD-worthy primitives of aggregation: CONTIGUOUS MASKED
//  REDUCTIONS (sum / min / max) of a column into a single accumulator.
//
//  WHERE THEY ARE USED. The GLOBAL (zero-key) aggregate is a pure reduction over
//  the whole input, so it runs through these kernels. The GROUPED (>=1 key) path
//  is an inherently-sequential scatter into per-group state (duplicate group ids
//  within a SIMD lane cannot be scatter-added safely/portably with Highway) — so
//  it stays scalar control flow, exactly like the WP-4 probe walk. See the WP
//  report for the rationale and the future-work note.
//
//  NULL HANDLING IS THE CALLER'S. By the time a reduction kernel runs, NULL
//  inputs have already been folded to the op's IDENTITY by the operator
//  (sum: 0 / 0.0; min: type-max / +inf; max: type-min / -inf), and the count of
//  NON-NULL inputs is obtained as agg_sum_i64 over a per-row 0/1 array. So these
//  kernels are pure reductions; the null/seen SEMANTICS live in the operator and
//  are the target of the WP-5 mutation self-test.
//
//  TWIN CONVENTION (simd/kernel_convention.h, D17). Each kernel exposes two free
//  functions with identical signatures and INDEPENDENT code in SEPARATE TUs:
//    *_vec    — Highway dynamic-dispatch path (ops/agg_kernels.cpp)
//    *_scalar — hand-written reference loop     (ops/agg_scalar.cpp)
//  tests/agg_scalar_vector_test.cpp proves they agree on every input including
//  the SIMD-tail boundary lengths. No vector width / ISA appears here.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::ops {

// SUM reductions. The integer accumulator is i64 (the SUM(I32)/SUM(I64) result
// type). n == 0 yields the identity (0 / 0.0).
std::int64_t agg_sum_i64_vec(const std::int64_t* v, std::size_t n);
std::int64_t agg_sum_i64_scalar(const std::int64_t* v, std::size_t n);
double agg_sum_f64_vec(const double* v, std::size_t n);
double agg_sum_f64_scalar(const double* v, std::size_t n);

// MIN / MAX reductions. Precondition: n >= 1 (the operator never reduces an
// empty batch). Null lanes must already be the op identity (so they never win).
std::int64_t agg_min_i64_vec(const std::int64_t* v, std::size_t n);
std::int64_t agg_min_i64_scalar(const std::int64_t* v, std::size_t n);
std::int64_t agg_max_i64_vec(const std::int64_t* v, std::size_t n);
std::int64_t agg_max_i64_scalar(const std::int64_t* v, std::size_t n);
double agg_min_f64_vec(const double* v, std::size_t n);
double agg_min_f64_scalar(const double* v, std::size_t n);
double agg_max_f64_vec(const double* v, std::size_t n);
double agg_max_f64_scalar(const double* v, std::size_t n);

}  // namespace qe::ops
