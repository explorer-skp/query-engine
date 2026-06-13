//  WP-1: TEST-ONLY planted-mutant kernels. See simd/validity_kernels_mutants.h.
//
//  Same Highway dynamic-dispatch shape as the real kernels in
//  simd/validity_kernels.cpp, but each carries one deliberately-broken tail. The
//  mutation self-test links this and shows the differential catches the break.
//  Built with -Wno-error (third-party Highway headers).

#include "simd/validity_kernels_mutants.h"

#include <bit>

#include "core/validity.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/validity_kernels_mutants.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::simd::mutant {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

std::size_t CountSetTailBugImpl(const std::uint64_t* w, std::size_t nbits) {
    const hn::ScalableTag<std::uint64_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t full = nbits / 64;
    auto vacc = hn::Zero(d);
    std::size_t i = 0;
    for (; i + lanes <= full; i += lanes) {
        vacc = hn::Add(vacc, hn::PopulationCount(hn::LoadU(d, w + i)));
    }
    std::uint64_t count = hn::ReduceSum(d, vacc);
    for (; i < full; ++i) {
        count += static_cast<std::uint64_t>(std::popcount(w[i]));
    }
    if (nbits & 63u) {
        // BUG: missing `& last_word_mask(nbits)` — padding bits are counted.
        count += static_cast<std::uint64_t>(std::popcount(w[full]));
    }
    return static_cast<std::size_t>(count);
}

bool AllValidTailBugImpl(const std::uint64_t* w, std::size_t nbits) {
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
    // BUG: the partial final word is never checked — a null there is missed.
    return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::simd::mutant
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::simd::mutant {

HWY_EXPORT(CountSetTailBugImpl);
HWY_EXPORT(AllValidTailBugImpl);

std::size_t count_set_vec_tailbug(const std::uint64_t* w, std::size_t nbits) {
    return HWY_DYNAMIC_DISPATCH(CountSetTailBugImpl)(w, nbits);
}

bool all_valid_vec_tailbug(const std::uint64_t* w, std::size_t nbits) {
    return HWY_DYNAMIC_DISPATCH(AllValidTailBugImpl)(w, nbits);
}

}  // namespace qe::simd::mutant
#endif  // HWY_ONCE
