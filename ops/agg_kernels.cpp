//  WP-5: VECTOR PATH for the masked-reduction kernels (see ops/agg_kernels.h).
//  Horizontal sum/min/max over a contiguous, already-identity-folded column via
//  Highway dynamic dispatch; the independently-written scalar twin is in
//  ops/agg_scalar.cpp. Width is whatever Highway selects at runtime — nothing is
//  hardcoded; a scalar coda handles the < one-vector remainder. Built with
//  -Wno-error (third-party Highway template headers).

#include "ops/agg_kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/agg_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::ops {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// --- SUM: vector accumulator, horizontal reduce, scalar tail -----------------
template <class T>
T SumImpl(const T* v, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    std::size_t k = 0;
    for (; k + lanes <= n; k += lanes) acc = hn::Add(acc, hn::LoadU(d, v + k));
    T s = hn::ReduceSum(d, acc);
    for (; k < n; ++k) s += v[k];
    return s;
}

// --- MIN / MAX: seed the accumulator from the first full block (Zero is NOT the
// identity for min/max), reduce, scalar tail. Pure scalar when n < one vector.
template <class T>
T MinImpl(const T* v, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    if (n < lanes) {
        T m = v[0];
        for (std::size_t k = 1; k < n; ++k)
            if (v[k] < m) m = v[k];
        return m;
    }
    auto acc = hn::LoadU(d, v);
    std::size_t k = lanes;
    for (; k + lanes <= n; k += lanes) acc = hn::Min(acc, hn::LoadU(d, v + k));
    T m = hn::ReduceMin(d, acc);
    for (; k < n; ++k)
        if (v[k] < m) m = v[k];
    return m;
}

template <class T>
T MaxImpl(const T* v, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    if (n < lanes) {
        T m = v[0];
        for (std::size_t k = 1; k < n; ++k)
            if (v[k] > m) m = v[k];
        return m;
    }
    auto acc = hn::LoadU(d, v);
    std::size_t k = lanes;
    for (; k + lanes <= n; k += lanes) acc = hn::Max(acc, hn::LoadU(d, v + k));
    T m = hn::ReduceMax(d, acc);
    for (; k < n; ++k)
        if (v[k] > m) m = v[k];
    return m;
}

std::int64_t SumI64(const std::int64_t* v, std::size_t n) { return SumImpl(v, n); }
double SumF64(const double* v, std::size_t n) { return SumImpl(v, n); }
std::int64_t MinI64(const std::int64_t* v, std::size_t n) { return MinImpl(v, n); }
std::int64_t MaxI64(const std::int64_t* v, std::size_t n) { return MaxImpl(v, n); }
double MinF64(const double* v, std::size_t n) { return MinImpl(v, n); }
double MaxF64(const double* v, std::size_t n) { return MaxImpl(v, n); }

}  // namespace HWY_NAMESPACE
}  // namespace qe::ops
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::ops {

HWY_EXPORT(SumI64);
HWY_EXPORT(SumF64);
HWY_EXPORT(MinI64);
HWY_EXPORT(MaxI64);
HWY_EXPORT(MinF64);
HWY_EXPORT(MaxF64);

std::int64_t agg_sum_i64_vec(const std::int64_t* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(SumI64)(v, n);
}
double agg_sum_f64_vec(const double* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(SumF64)(v, n);
}
std::int64_t agg_min_i64_vec(const std::int64_t* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(MinI64)(v, n);
}
std::int64_t agg_max_i64_vec(const std::int64_t* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(MaxI64)(v, n);
}
double agg_min_f64_vec(const double* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(MinF64)(v, n);
}
double agg_max_f64_vec(const double* v, std::size_t n) {
    return HWY_DYNAMIC_DISPATCH(MaxF64)(v, n);
}

}  // namespace qe::ops
#endif  // HWY_ONCE
