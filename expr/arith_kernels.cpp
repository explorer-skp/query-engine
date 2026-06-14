//  WP-2: VECTOR PATH for the arithmetic kernels (see expr/kernels.h).
//
//  The only-place-Highway-appears half of the arith twin (scalar twin:
//  expr/arith_scalar.cpp). foreach_target + HWY_DYNAMIC_DISPATCH, same pattern as
//  simd/validity_kernels.cpp. Built with -Wno-error (third-party Highway headers).
//
//  Integer +,-,* are vectorized with Highway's non-saturating ops, which wrap at
//  the hardware level (two's complement) — no C++ signed-overflow UB. The scalar
//  TAIL uses unsigned-intermediate wraps for the same reason. Integer /,% have NO
//  portable SIMD form (no SIMD integer-division instruction on NEON/AVX), so they
//  run a scalar guarded loop here — documented, like simd/gather8_scalar and the
//  WP-1 convention. Float +,-,*,/ are vectorized; float % (fmod) is scalar.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core/types.h"
#include "expr/expr.h"
#include "expr/kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "expr/arith_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::expr {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
T uwrap_add(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}
template <typename T>
T uwrap_sub(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}
template <typename T>
T uwrap_mul(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}

template <typename T>
void AddT(const T* a, const T* b, T* out, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>)
            out[i] = a[i] + b[i];
        else
            out[i] = uwrap_add(a[i], b[i]);
    }
}

template <typename T>
void SubT(const T* a, const T* b, T* out, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>)
            out[i] = a[i] - b[i];
        else
            out[i] = uwrap_sub(a[i], b[i]);
    }
}

template <typename T>
void MulT(const T* a, const T* b, T* out, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>)
            out[i] = a[i] * b[i];
        else
            out[i] = uwrap_mul(a[i], b[i]);
    }
}

// Float division is vectorized; integer division has no SIMD form (scalar guard).
template <typename T>
void DivT(const T* a, const T* b, T* out, std::size_t n) {
    if constexpr (std::is_floating_point_v<T>) {
        const hn::ScalableTag<T> d;
        const std::size_t lanes = hn::Lanes(d);
        std::size_t i = 0;
        for (; i + lanes <= n; i += lanes)
            hn::StoreU(hn::Div(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d,
                       out + i);
        for (; i < n; ++i) out[i] = a[i] / b[i];
    } else {
        using U = std::make_unsigned_t<T>;
        for (std::size_t i = 0; i < n; ++i) {
            const T bi = b[i];
            if (bi == 0)
                out[i] = 0;  // NULL'd by eval
            else if (bi == static_cast<T>(-1))
                out[i] = static_cast<T>(U{0} - static_cast<U>(a[i]));  // -a (wrap)
            else
                out[i] = static_cast<T>(a[i] / bi);
        }
    }
}

template <typename T>
void ModT(const T* a, const T* b, T* out, std::size_t n) {
    if constexpr (std::is_floating_point_v<T>) {
        for (std::size_t i = 0; i < n; ++i) out[i] = std::fmod(a[i], b[i]);
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const T bi = b[i];
            if (bi == 0 || bi == static_cast<T>(-1))
                out[i] = 0;
            else
                out[i] = static_cast<T>(a[i] % bi);
        }
    }
}

template <typename T>
void DispatchT(ArithOp op, const void* a, const void* b, void* out,
               std::size_t n) {
    const T* pa = static_cast<const T*>(a);
    const T* pb = static_cast<const T*>(b);
    T* po = static_cast<T*>(out);
    switch (op) {
        case ArithOp::Add: AddT(pa, pb, po, n); break;
        case ArithOp::Sub: SubT(pa, pb, po, n); break;
        case ArithOp::Mul: MulT(pa, pb, po, n); break;
        case ArithOp::Div: DivT(pa, pb, po, n); break;
        case ArithOp::Mod: ModT(pa, pb, po, n); break;
    }
}

void ArithImpl(ArithOp op, Type t, const void* a, const void* b, void* out,
               std::size_t n) {
    switch (t) {
        case Type::I32: DispatchT<std::int32_t>(op, a, b, out, n); break;
        case Type::I64: DispatchT<std::int64_t>(op, a, b, out, n); break;
        case Type::F64: DispatchT<double>(op, a, b, out, n); break;
        case Type::BOOL:
        case Type::TS:
            break;  // builder never produces arithmetic on these
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::expr
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::expr {

HWY_EXPORT(ArithImpl);

void arith_vec(ArithOp op, Type t, const void* a, const void* b, void* out,
               std::size_t n) {
    HWY_DYNAMIC_DISPATCH(ArithImpl)(op, t, a, b, out, n);
}

}  // namespace qe::expr
#endif  // HWY_ONCE
