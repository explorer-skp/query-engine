//  WP-2: VECTOR PATH for the cast kernels (see expr/kernels.h).
//
//  Numeric<->numeric conversions among I32, I64/TS, F64 are vectorized with
//  Highway promote/demote/convert (32-bit lanes are loaded/stored through a
//  Rebind to the 64-bit lane count, the WP-1 gather64 idiom). float->int rounds
//  half away from zero and zeroes out-of-range/non-finite lanes so the produced
//  VALUE matches the scalar twin (expr/cast_scalar.cpp) byte-for-byte at EVERY
//  lane (eval NULLs the same lanes via expr/cast_round.h). int->int narrowing
//  truncates low bits (matching the twin's static_cast), eval NULLs out-of-range.
//
//  BOOL conversions (to/from BOOL) have no clean uniform SIMD form (1-byte lanes
//  vs wide numeric lanes) and use an independently-written per-lane loop here —
//  documented, not an omission, mirroring WP-1's gather8_scalar decision.
//  Built with -Wno-error (third-party Highway headers).

#include <cstddef>
#include <cstdint>

#include "core/types.h"
#include "expr/cast_round.h"
#include "expr/kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "expr/cast_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::expr {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// rep(t): the physical lane representation a numeric Type casts through.
enum class Rep { I32, I64, F64, BOOL };
Rep rep_of(Type t) {
    switch (t) {
        case Type::I32: return Rep::I32;
        case Type::I64:
        case Type::TS: return Rep::I64;
        case Type::F64: return Rep::F64;
        case Type::BOOL: return Rep::BOOL;
    }
    return Rep::I64;
}

void I32toI64(const std::int32_t* a, std::int64_t* out, std::size_t n) {
    const hn::ScalableTag<std::int64_t> d64;
    const hn::Rebind<std::int32_t, decltype(d64)> d32;
    const std::size_t S = hn::Lanes(d64);
    std::size_t i = 0;
    for (; i + S <= n; i += S)
        hn::StoreU(hn::PromoteTo(d64, hn::LoadU(d32, a + i)), d64, out + i);
    for (; i < n; ++i) out[i] = a[i];
}

void I32toF64(const std::int32_t* a, double* out, std::size_t n) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<std::int32_t, decltype(dd)> d32;
    const std::size_t S = hn::Lanes(dd);
    std::size_t i = 0;
    for (; i + S <= n; i += S)
        hn::StoreU(hn::PromoteTo(dd, hn::LoadU(d32, a + i)), dd, out + i);
    for (; i < n; ++i) out[i] = static_cast<double>(a[i]);
}

void I64toI32(const std::int64_t* a, std::int32_t* out, std::size_t n) {
    const hn::ScalableTag<std::int64_t> d64;
    const hn::RebindToUnsigned<decltype(d64)> du64;
    const hn::Rebind<std::uint32_t, decltype(d64)> du32;
    const hn::Rebind<std::int32_t, decltype(d64)> d32;
    const std::size_t S = hn::Lanes(d64);
    std::size_t i = 0;
    for (; i + S <= n; i += S) {
        const auto t = hn::TruncateTo(du32, hn::BitCast(du64, hn::LoadU(d64, a + i)));
        hn::StoreU(hn::BitCast(d32, t), d32, out + i);
    }
    for (; i < n; ++i) out[i] = static_cast<std::int32_t>(a[i]);
}

void I64toF64(const std::int64_t* a, double* out, std::size_t n) {
    const hn::ScalableTag<std::int64_t> d64;
    const hn::ScalableTag<double> dd;
    const std::size_t S = hn::Lanes(d64);
    std::size_t i = 0;
    for (; i + S <= n; i += S)
        hn::StoreU(hn::ConvertTo(dd, hn::LoadU(d64, a + i)), dd, out + i);
    for (; i < n; ++i) out[i] = static_cast<double>(a[i]);
}

// Round half away from zero, vectorized — must match std::round / cast_round
// for EVERY finite double, including large-magnitude integral values.
//
// PINNED BUG (do not "simplify" back): the naive Trunc(x + copysign(0.5,x)) is
// WRONG for |x| >= 2^52, where the ULP is >= 1 and adding 0.5 rounds an
// already-integral value up to the next integer (e.g. -7901924385117227 ->
// -7901924385117228). Instead, truncate toward zero and add ±1 only when the
// exact fractional part is >= 0.5. For |x| >= 2^52 the fractional part is exactly
// 0, so the result is x unchanged — correct. For |x| < 2^52, x - Trunc(x) and
// Trunc(x) ± 1 are all exactly representable, so this matches std::round.
template <class D>
auto RoundAway(D dd, decltype(hn::Zero(D())) x) {
    const auto t = hn::Trunc(x);              // toward zero (integral)
    const auto frac = hn::Sub(x, t);          // exact fractional part
    const auto bump = hn::IfThenElseZero(
        hn::Ge(hn::Abs(frac), hn::Set(dd, 0.5)), hn::CopySign(hn::Set(dd, 1.0), x));
    return hn::Add(t, bump);
}

void F64toI32(const double* a, std::int32_t* out, std::size_t n, double lo,
              double hi) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<std::int32_t, decltype(dd)> d32;
    const std::size_t S = hn::Lanes(dd);
    std::size_t i = 0;
    for (; i + S <= n; i += S) {
        const auto x = hn::LoadU(dd, a + i);
        const auto r = RoundAway(dd, x);
        const auto fits = hn::And(hn::Ge(r, hn::Set(dd, lo)), hn::Lt(r, hn::Set(dd, hi)));
        const auto r0 = hn::IfThenElseZero(fits, r);
        hn::StoreU(hn::DemoteTo(d32, r0), d32, out + i);
    }
    for (; i < n; ++i)
        out[i] = f64_fits_int(a[i], Type::I32)
                     ? static_cast<std::int32_t>(cast_round(a[i]))
                     : 0;
}

void F64toI64(const double* a, std::int64_t* out, std::size_t n, double lo,
              double hi) {
    const hn::ScalableTag<double> dd;
    const hn::ScalableTag<std::int64_t> d64;
    const std::size_t S = hn::Lanes(dd);
    std::size_t i = 0;
    for (; i + S <= n; i += S) {
        const auto x = hn::LoadU(dd, a + i);
        const auto r = RoundAway(dd, x);
        const auto fits = hn::And(hn::Ge(r, hn::Set(dd, lo)), hn::Lt(r, hn::Set(dd, hi)));
        const auto r0 = hn::IfThenElseZero(fits, r);
        hn::StoreU(hn::ConvertTo(d64, r0), d64, out + i);
    }
    for (; i < n; ++i)
        out[i] = f64_fits_int(a[i], Type::I64)
                     ? static_cast<std::int64_t>(cast_round(a[i]))
                     : 0;
}

// Independently-written per-lane BOOL conversions (documented scalar corner).
std::int64_t read_int_lane(Type from, const void* in, std::size_t i) {
    switch (from) {
        case Type::I32: return static_cast<const std::int32_t*>(in)[i];
        case Type::I64:
        case Type::TS: return static_cast<const std::int64_t*>(in)[i];
        case Type::BOOL: return static_cast<const std::uint8_t*>(in)[i];
        default: return 0;
    }
}

void ToBool(Type from, const void* in, std::uint8_t* out, std::size_t n) {
    if (from == Type::F64) {
        const double* d = static_cast<const double*>(in);
        for (std::size_t i = 0; i < n; ++i) out[i] = (d[i] != 0.0) ? 1 : 0;
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (read_int_lane(from, in, i) != 0) ? 1 : 0;
    }
}

void BoolTo(Type to, const std::uint8_t* in, void* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t v = in[i] ? 1 : 0;
        switch (to) {
            case Type::I32: static_cast<std::int32_t*>(out)[i] = static_cast<std::int32_t>(v); break;
            case Type::I64:
            case Type::TS: static_cast<std::int64_t*>(out)[i] = v; break;
            case Type::F64: static_cast<double*>(out)[i] = static_cast<double>(v); break;
            default: break;
        }
    }
}

template <typename T>
void CopyT(const void* in, void* out, std::size_t n) {
    const hn::ScalableTag<T> d;
    const std::size_t S = hn::Lanes(d);
    const T* a = static_cast<const T*>(in);
    T* o = static_cast<T*>(out);
    std::size_t i = 0;
    for (; i + S <= n; i += S) hn::StoreU(hn::LoadU(d, a + i), d, o + i);
    for (; i < n; ++i) o[i] = a[i];
}

void CastImpl(Type from, Type to, const void* in, void* out, std::size_t n) {
    const Rep rf = rep_of(from), rt = rep_of(to);
    if (rf == Rep::BOOL) {
        BoolTo(to, static_cast<const std::uint8_t*>(in), out, n);
        return;
    }
    if (rt == Rep::BOOL) {
        ToBool(from, in, static_cast<std::uint8_t*>(out), n);
        return;
    }
    if (rf == rt) {  // I64<->TS (and any same-rep): bit copy
        if (rf == Rep::I32) CopyT<std::int32_t>(in, out, n);
        else CopyT<std::int64_t>(in, out, n);  // i64 and f64 are both 8 bytes
        return;
    }
    const auto* i32 = static_cast<const std::int32_t*>(in);
    const auto* i64 = static_cast<const std::int64_t*>(in);
    const auto* f64 = static_cast<const double*>(in);
    switch (rf) {
        case Rep::I32:
            if (rt == Rep::I64) I32toI64(i32, static_cast<std::int64_t*>(out), n);
            else I32toF64(i32, static_cast<double*>(out), n);  // rt == F64
            break;
        case Rep::I64:
            if (rt == Rep::I32) I64toI32(i64, static_cast<std::int32_t*>(out), n);
            else I64toF64(i64, static_cast<double*>(out), n);  // rt == F64
            break;
        case Rep::F64:
            if (rt == Rep::I32)
                F64toI32(f64, static_cast<std::int32_t*>(out), n, -2147483648.0,
                         2147483648.0);
            else  // rt == I64 (or TS)
                F64toI64(f64, static_cast<std::int64_t*>(out), n,
                         -9223372036854775808.0, 9223372036854775808.0);
            break;
        case Rep::BOOL:
            break;  // handled above
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::expr
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::expr {

HWY_EXPORT(CastImpl);

void cast_vec(Type from, Type to, const void* in, void* out, std::size_t n) {
    HWY_DYNAMIC_DISPATCH(CastImpl)(from, to, in, out, n);
}

}  // namespace qe::expr
#endif  // HWY_ONCE
