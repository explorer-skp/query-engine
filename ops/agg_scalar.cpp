//  WP-5: SCALAR TWIN for the masked-reduction kernels (see ops/agg_kernels.h). A
//  plain, Highway-free reference — INDEPENDENT code from the vector path in
//  ops/agg_kernels.cpp (D17: collapsing them would make scalar==vector prove
//  nothing). One straight loop per reduction; the scalar==vector differential
//  asserts it agrees with the vector path on every input, including the SIMD-tail
//  boundary lengths.

#include <cmath>
#include <limits>

#include "ops/agg_kernels.h"

namespace qe::ops {

std::int64_t agg_sum_i64_scalar(const std::int64_t* v, std::size_t n) {
    std::int64_t s = 0;
    for (std::size_t k = 0; k < n; ++k) s += v[k];
    return s;
}

double agg_sum_f64_scalar(const double* v, std::size_t n) {
    double s = 0.0;
    for (std::size_t k = 0; k < n; ++k) s += v[k];
    return s;
}

std::int64_t agg_min_i64_scalar(const std::int64_t* v, std::size_t n) {
    std::int64_t m = v[0];
    for (std::size_t k = 1; k < n; ++k)
        if (v[k] < m) m = v[k];
    return m;
}

std::int64_t agg_max_i64_scalar(const std::int64_t* v, std::size_t n) {
    std::int64_t m = v[0];
    for (std::size_t k = 1; k < n; ++k)
        if (v[k] > m) m = v[k];
    return m;
}

// F64 MIN/MAX use the NaN-greatest TOTAL order (DuckDB semantics, audit C2):
// MIN is NaN only when every element is NaN; MAX is NaN when any element is.
// Plain running `<`/`>` loops silently drop NaNs position-dependently. This is
// the independent scalar formulation of the policy — the vector path expresses
// it differently (NaN-lane masking); scalar==vector proves they agree.
double agg_min_f64_scalar(const double* v, std::size_t n) {
    double m = std::numeric_limits<double>::quiet_NaN();
    bool seen_real = false;
    for (std::size_t k = 0; k < n; ++k) {
        if (std::isnan(v[k])) continue;
        if (!seen_real || v[k] < m) m = v[k];
        seen_real = true;
    }
    return m;  // all-NaN (or the folded-identity slots) => NaN
}

double agg_max_f64_scalar(const double* v, std::size_t n) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
        if (std::isnan(v[k]))
            return std::numeric_limits<double>::quiet_NaN();  // NaN is greatest
        if (v[k] > m) m = v[k];
    }
    return m;
}

}  // namespace qe::ops
