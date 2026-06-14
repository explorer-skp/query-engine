//  WP-4: VECTOR PATH for the hash-combine kernel (see ops/hash_kernels.h). The
//  splitmix64 finalizer over u64 lanes via Highway dynamic dispatch; the scalar
//  twin lives in ops/hash_scalar.cpp. The two 64-bit multiplies use Highway's
//  portable u64 operator* (native on AVX-512, emulated from 32-bit MulEven on
//  NEON) — so ONE source compiles to both ISAs with no width hardcoded. A scalar
//  coda handles the < one-vector remainder. Built with -Wno-error (third-party
//  Highway headers).

#include "ops/hash_kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/hash_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::ops {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// splitmix64 finalizer over a u64 lane vector — the SAME arithmetic the scalar
// twin (ops/hash_scalar.cpp) performs lane-by-lane.
template <class D, class V>
V Mix64(D d, V x) {
    const auto c1 = hn::Set(d, 0xbf58476d1ce4e5b9ull);
    const auto c2 = hn::Set(d, 0x94d049bb133111ebull);
    x = hn::Xor(x, hn::ShiftRight<30>(x));
    x = hn::Mul(x, c1);
    x = hn::Xor(x, hn::ShiftRight<27>(x));
    x = hn::Mul(x, c2);
    x = hn::Xor(x, hn::ShiftRight<31>(x));
    return x;
}

void HashCombineImpl(std::uint64_t* acc, const std::uint64_t* words,
                     std::size_t n) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t k = 0;
    for (; k + lanes <= n; k += lanes) {
        const auto w = hn::LoadU(d, words + k);
        const auto a = hn::LoadU(d, acc + k);
        const auto mixed = Mix64(d, hn::Xor(a, Mix64(d, w)));
        hn::StoreU(mixed, d, acc + k);
    }
    // Scalar coda — identical formula on the remainder lanes.
    for (; k < n; ++k) {
        std::uint64_t x = words[k];
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        std::uint64_t y = acc[k] ^ x;
        y ^= y >> 30;
        y *= 0xbf58476d1ce4e5b9ull;
        y ^= y >> 27;
        y *= 0x94d049bb133111ebull;
        y ^= y >> 31;
        acc[k] = y;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::ops
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::ops {

HWY_EXPORT(HashCombineImpl);

void hash_combine_vec(std::uint64_t* acc, const std::uint64_t* words,
                      std::size_t n) {
    HWY_DYNAMIC_DISPATCH(HashCombineImpl)(acc, words, n);
}

}  // namespace qe::ops
#endif  // HWY_ONCE
