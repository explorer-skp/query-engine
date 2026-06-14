//  WP-2: TEST-ONLY planted-mutant kernels. See expr/expr_kernels_mutants.h.
//  Each has one deliberate bug; the mutation self-test shows it is caught. Built
//  with -Wno-error (third-party Highway headers).

#include "expr/expr_kernels_mutants.h"

#include <cstring>

#include "core/validity.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "expr/expr_kernels_mutants.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::expr::mutant {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void CmpLtTailBugImpl(const std::int32_t* a, const std::int32_t* b,
                      std::uint8_t* out, std::size_t n) {
    const hn::ScalableTag<std::int32_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto one = hn::Set(d, 1);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto m = hn::Lt(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        const auto v01 = hn::IfThenElseZero(m, one);
        HWY_ALIGN std::int32_t tmp[hn::MaxLanes(d)];
        hn::Store(v01, d, tmp);
        for (std::size_t l = 0; l < lanes; ++l)
            out[i + l] = static_cast<std::uint8_t>(tmp[l]);
    }
    // BUG: the scalar tail (i < n) is missing — tail lanes are never written.
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::expr::mutant
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::expr::mutant {

HWY_EXPORT(CmpLtTailBugImpl);

void cmp_lt_i32_tailbug_vec(const std::int32_t* a, const std::int32_t* b,
                            std::uint8_t* out, std::size_t n) {
    HWY_DYNAMIC_DISPATCH(CmpLtTailBugImpl)(a, b, out, n);
}

void logic_and_twovalued(const std::uint8_t* a, const std::uint8_t* b,
                         std::uint8_t* out, std::size_t n) {
    // Tri-state 0=F,1=N,2=T. BUG: plain null-if-any instead of Kleene — FALSE∧
    // NULL becomes NULL(1) instead of FALSE(0). (Correct AND is min(a,b).)
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] == 1 || b[i] == 1)
            out[i] = 1;                          // null-if-any (WRONG for F∧N)
        else
            out[i] = (a[i] == 2 && b[i] == 2) ? 2 : 0;
    }
}

void propagate_nulls_and_allvalidbug(OwnedColumn& out, const Column& a,
                                     const Column& b) {
    const std::size_t n = out.len();
    // BUG: short-circuits when EITHER operand is all-valid (correct: only when
    // BOTH are), so the other operand's nulls are silently dropped.
    if (a.all_valid || b.all_valid) return;
    out.ensure_validity();
    std::uint64_t* ov = out.mutable_validity();
    const std::size_t w = validity::words(n);
    if (a.validity == nullptr) {
        std::memcpy(ov, b.validity, w * sizeof(std::uint64_t));
    } else if (b.validity == nullptr) {
        std::memcpy(ov, a.validity, w * sizeof(std::uint64_t));
    } else {
        for (std::size_t k = 0; k < w; ++k) ov[k] = a.validity[k] & b.validity[k];
    }
    out.refresh_all_valid();
}

}  // namespace qe::expr::mutant
#endif  // HWY_ONCE
