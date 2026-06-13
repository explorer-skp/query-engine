//  WP-1: VECTOR PATH for the validity-bitmap kernels (see simd/validity_kernels.h).
//
//  This is the only-place-Highway-appears half of the twin. It uses Highway's
//  foreach_target machinery to compile one variant per supported target and
//  HWY_DYNAMIC_DISPATCH to pick the right one at runtime — the same pattern as
//  tools/hwy_smoke.cpp. The scalar twins live in simd/validity_scalar.cpp and are
//  deliberately independent code. Built with -Wno-error (third-party Highway
//  template headers).
//
//  Tail discipline (the thing a SIMD bug gets wrong): every kernel processes
//  whole 64-bit words in vectors, then handles the remainder words AND the
//  partial final word (masked via qe::validity::last_word_mask) in a scalar
//  coda. Dropping or mis-masking that coda is exactly the planted mutation in
//  simd/validity_kernels_mutants.cpp.

#include "simd/validity_kernels.h"

#include <bit>

#include "core/validity.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/validity_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

std::size_t CountSetImpl(const std::uint64_t* w, std::size_t nbits) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t full = nbits / 64;  // fully-covered words
    auto vacc = hn::Zero(d);
    std::size_t i = 0;
    for (; i + lanes <= full; i += lanes) {
        vacc = hn::Add(vacc, hn::PopulationCount(hn::LoadU(d, w + i)));
    }
    std::uint64_t count = hn::ReduceSum(d, vacc);
    for (; i < full; ++i) {  // remainder whole words
        count += static_cast<std::uint64_t>(std::popcount(w[i]));
    }
    if (nbits & 63u) {  // partial final word: mask, then count
        const std::uint64_t m = qe::validity::last_word_mask(nbits);
        count += static_cast<std::uint64_t>(std::popcount(w[full] & m));
    }
    return static_cast<std::size_t>(count);
}

bool AllValidImpl(const std::uint64_t* w, std::size_t nbits) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto ones = hn::Set(d, ~0ull);
    const std::size_t full = nbits / 64;
    std::size_t i = 0;
    for (; i + lanes <= full; i += lanes) {
        if (!hn::AllTrue(d, hn::Eq(hn::LoadU(d, w + i), ones))) return false;
    }
    for (; i < full; ++i) {
        if (w[i] != ~0ull) return false;
    }
    if (nbits & 63u) {
        const std::uint64_t m = qe::validity::last_word_mask(nbits);
        if ((w[full] & m) != m) return false;
    }
    return true;
}

void BitAndImpl(const std::uint64_t* a, const std::uint64_t* b,
                std::uint64_t* out, std::size_t nbits) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t n = qe::validity::words(nbits);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        hn::StoreU(hn::And(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d,
                   out + i);
    }
    for (; i < n; ++i) out[i] = a[i] & b[i];
}

void BitOrImpl(const std::uint64_t* a, const std::uint64_t* b,
               std::uint64_t* out, std::size_t nbits) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t n = qe::validity::words(nbits);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        hn::StoreU(hn::Or(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    }
    for (; i < n; ++i) out[i] = a[i] | b[i];
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::simd {

HWY_EXPORT(CountSetImpl);
HWY_EXPORT(AllValidImpl);
HWY_EXPORT(BitAndImpl);
HWY_EXPORT(BitOrImpl);

std::size_t count_set_vec(const std::uint64_t* w, std::size_t nbits) {
    return HWY_DYNAMIC_DISPATCH(CountSetImpl)(w, nbits);
}

bool all_valid_vec(const std::uint64_t* w, std::size_t nbits) {
    return HWY_DYNAMIC_DISPATCH(AllValidImpl)(w, nbits);
}

bool any_null_vec(const std::uint64_t* w, std::size_t nbits) {
    return !HWY_DYNAMIC_DISPATCH(AllValidImpl)(w, nbits);
}

void bit_and_vec(const std::uint64_t* a, const std::uint64_t* b,
                 std::uint64_t* out, std::size_t nbits) {
    HWY_DYNAMIC_DISPATCH(BitAndImpl)(a, b, out, nbits);
}

void bit_or_vec(const std::uint64_t* a, const std::uint64_t* b,
                std::uint64_t* out, std::size_t nbits) {
    HWY_DYNAMIC_DISPATCH(BitOrImpl)(a, b, out, nbits);
}

}  // namespace qe::simd
#endif  // HWY_ONCE
