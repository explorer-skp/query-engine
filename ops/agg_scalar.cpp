//  WP-5: SCALAR TWIN for the masked-reduction kernels (see ops/agg_kernels.h). A
//  plain, Highway-free reference — INDEPENDENT code from the vector path in
//  ops/agg_kernels.cpp (D17: collapsing them would make scalar==vector prove
//  nothing). One straight loop per reduction; the scalar==vector differential
//  asserts it agrees with the vector path on every input, including the SIMD-tail
//  boundary lengths.

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

double agg_min_f64_scalar(const double* v, std::size_t n) {
    double m = v[0];
    for (std::size_t k = 1; k < n; ++k)
        if (v[k] < m) m = v[k];
    return m;
}

double agg_max_f64_scalar(const double* v, std::size_t n) {
    double m = v[0];
    for (std::size_t k = 1; k < n; ++k)
        if (v[k] > m) m = v[k];
    return m;
}

}  // namespace qe::ops
