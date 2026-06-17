//  WP-14: VECTOR PATH for the zig-zag-decode kernel (see tsx/compress_kernels.h).
//  The only-place-Highway-appears half of the twin, built exactly like the WP-1
//  validity kernels (simd/validity_kernels.cpp): foreach_target compiles one
//  variant per supported ISA and HWY_DYNAMIC_DISPATCH selects it at runtime. No
//  vector width / ISA appears in the body — Highway's ScalableTag/Lanes drive the
//  loop. The scalar twin lives in tsx/compress_scalar.cpp as separate code. Built
//  with -Wno-error (third-party Highway template headers).
//
//  Tail discipline: whole-vector lanes first, then a scalar coda for the remainder
//  — the same shape every kernel here follows; the coda is plain unsigned bit ops.

#include "tsx/compress_kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tsx/compress_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::tsx {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// out[k] = (zz[k] >> 1) ^ (0 - (zz[k] & 1)).  ShiftRight is the LOGICAL shift on
// the unsigned lane; the sign mask is Sub(0, lsb), which wraps to all-ones when the
// low bit is set (two's complement) — matching the scalar `0ull - (v & 1)`.
void UnZigZagImpl(const std::uint64_t* zz, std::size_t n, std::uint64_t* out) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto one = hn::Set(d, 1ull);
    const auto zero = hn::Zero(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto v = hn::LoadU(d, zz + i);
        const auto shifted = hn::ShiftRight<1>(v);
        const auto mask = hn::Sub(zero, hn::And(v, one));  // 0 - lsb => 0 or ~0
        hn::StoreU(hn::Xor(shifted, mask), d, out + i);
    }
    for (; i < n; ++i) {
        const std::uint64_t v = zz[i];
        out[i] = (v >> 1) ^ (0ull - (v & 1ull));
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::tsx
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::tsx {

HWY_EXPORT(UnZigZagImpl);

void unzigzag_vec(const std::uint64_t* zz, std::size_t n, std::uint64_t* out) {
    HWY_DYNAMIC_DISPATCH(UnZigZagImpl)(zz, n, out);
}

}  // namespace qe::tsx
#endif  // HWY_ONCE
