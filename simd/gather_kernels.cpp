//  WP-1: VECTOR PATH for the selection-vector gather kernels (see
//  simd/gather_kernels.h). Uses Highway's GatherIndex through dynamic dispatch;
//  the scalar twins live in simd/gather_scalar.cpp. Built with -Wno-error
//  (third-party Highway headers).
//
//  For 64-bit lanes the uint32 selection indices are widened to int64 before the
//  gather (Highway's index lanes match the value lane width). A scalar coda
//  handles the < one-vector remainder.

#include "simd/gather_kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/gather_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void Gather32Impl(const std::uint32_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint32_t* out) {
    const hn::ScalableTag<std::uint32_t> d;
    const hn::RebindToSigned<decltype(d)> di;  // int32 index lanes
    const std::size_t lanes = hn::Lanes(d);
    std::size_t k = 0;
    if (idx != nullptr) {
        for (; k + lanes <= n; k += lanes) {
            const auto iv = hn::BitCast(di, hn::LoadU(d, idx + k));
            hn::StoreU(hn::GatherIndex(d, src, iv), d, out + k);
        }
        for (; k < n; ++k) out[k] = src[idx[k]];
    } else {
        for (; k + lanes <= n; k += lanes) {
            hn::StoreU(hn::LoadU(d, src + k), d, out + k);
        }
        for (; k < n; ++k) out[k] = src[k];
    }
}

void Gather64Impl(const std::uint64_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint64_t* out) {
    const hn::ScalableTag<std::uint64_t> d;
    const hn::RebindToSigned<decltype(d)> di;        // int64 index lanes
    const hn::Rebind<std::uint32_t, decltype(d)> du32;  // uint32, d-many lanes
    const std::size_t lanes = hn::Lanes(d);
    std::size_t k = 0;
    if (idx != nullptr) {
        for (; k + lanes <= n; k += lanes) {
            const auto i32 = hn::LoadU(du32, idx + k);
            const auto i64 = hn::PromoteTo(d, i32);  // uint32 -> uint64
            const auto iv = hn::BitCast(di, i64);    // -> int64 indices
            hn::StoreU(hn::GatherIndex(d, src, iv), d, out + k);
        }
        for (; k < n; ++k) out[k] = src[idx[k]];
    } else {
        for (; k + lanes <= n; k += lanes) {
            hn::StoreU(hn::LoadU(d, src + k), d, out + k);
        }
        for (; k < n; ++k) out[k] = src[k];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::simd {

HWY_EXPORT(Gather32Impl);
HWY_EXPORT(Gather64Impl);

void gather32_vec(const std::uint32_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint32_t* out) {
    HWY_DYNAMIC_DISPATCH(Gather32Impl)(src, idx, n, out);
}

void gather64_vec(const std::uint64_t* src, const std::uint32_t* idx,
                  std::size_t n, std::uint64_t* out) {
    HWY_DYNAMIC_DISPATCH(Gather64Impl)(src, idx, n, out);
}

}  // namespace qe::simd
#endif  // HWY_ONCE
