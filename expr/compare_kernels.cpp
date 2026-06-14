//  WP-2: VECTOR PATH for the comparison kernels (see expr/kernels.h).
//
//  Highway compares produce a mask per T-lane; we select 1/0 into a T-lane vector
//  and write the low byte of each lane to the BOOL (1-byte-per-value) output. The
//  scalar coda handles the < one-vector remainder. Scalar twin:
//  expr/compare_scalar.cpp. Built with -Wno-error (third-party Highway headers).

#include <cstddef>
#include <cstdint>

#include "core/types.h"
#include "expr/expr.h"
#include "expr/kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "expr/compare_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::expr {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
bool cmp_one(CmpOp op, T x, T y) {
    switch (op) {
        case CmpOp::Lt: return x < y;
        case CmpOp::Le: return x <= y;
        case CmpOp::Gt: return x > y;
        case CmpOp::Ge: return x >= y;
        case CmpOp::Eq: return x == y;
        case CmpOp::Ne: return x != y;
    }
    return false;
}

template <typename T>
void CmpT(CmpOp op, const T* a, const T* b, std::uint8_t* out, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto one = hn::Set(d, static_cast<T>(1));
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        decltype(hn::Eq(va, vb)) m;
        switch (op) {
            case CmpOp::Lt: m = hn::Lt(va, vb); break;
            case CmpOp::Le: m = hn::Le(va, vb); break;
            case CmpOp::Gt: m = hn::Gt(va, vb); break;
            case CmpOp::Ge: m = hn::Ge(va, vb); break;
            case CmpOp::Eq: m = hn::Eq(va, vb); break;
            case CmpOp::Ne: m = hn::Ne(va, vb); break;
        }
        const auto v01 = hn::IfThenElseZero(m, one);
        HWY_ALIGN T tmp[hn::MaxLanes(d)];
        hn::Store(v01, d, tmp);
        for (std::size_t l = 0; l < lanes; ++l)
            out[i + l] = static_cast<std::uint8_t>(tmp[l]);
    }
    for (; i < n; ++i) out[i] = cmp_one(op, a[i], b[i]) ? 1 : 0;
}

void CmpImpl(CmpOp op, Type t, const void* a, const void* b, std::uint8_t* out,
             std::size_t n) {
    switch (t) {
        case Type::I32:
            CmpT<std::int32_t>(op, static_cast<const std::int32_t*>(a),
                               static_cast<const std::int32_t*>(b), out, n);
            break;
        case Type::I64:
        case Type::TS:
            CmpT<std::int64_t>(op, static_cast<const std::int64_t*>(a),
                               static_cast<const std::int64_t*>(b), out, n);
            break;
        case Type::F64:
            CmpT<double>(op, static_cast<const double*>(a),
                         static_cast<const double*>(b), out, n);
            break;
        case Type::BOOL:
            CmpT<std::uint8_t>(op, static_cast<const std::uint8_t*>(a),
                               static_cast<const std::uint8_t*>(b), out, n);
            break;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::expr
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::expr {

HWY_EXPORT(CmpImpl);

void cmp_vec(CmpOp op, Type t, const void* a, const void* b, std::uint8_t* out,
             std::size_t n) {
    HWY_DYNAMIC_DISPATCH(CmpImpl)(op, t, a, b, out, n);
}

}  // namespace qe::expr
#endif  // HWY_ONCE
