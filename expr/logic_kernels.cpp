//  WP-2: VECTOR PATH for the three-valued logic kernels (see expr/kernels.h).
//
//  Tri-state bytes (0=F, 1=N, 2=T) make Kleene AND = Min, OR = Max, NOT = 2 - x.
//  The scalar twin (expr/logic_scalar.cpp) implements the SAME truth tables via
//  explicit branches — independent code. Built with -Wno-error (Highway headers).

#include <cstddef>
#include <cstdint>

#include "expr/kernels.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "expr/logic_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::expr {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void AndImpl(const std::uint8_t* a, const std::uint8_t* b, std::uint8_t* out,
             std::size_t n) {
    const hn::ScalableTag<std::uint8_t> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Min(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) out[i] = a[i] < b[i] ? a[i] : b[i];
}

void OrImpl(const std::uint8_t* a, const std::uint8_t* b, std::uint8_t* out,
            std::size_t n) {
    const hn::ScalableTag<std::uint8_t> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Max(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
}

void NotImpl(const std::uint8_t* a, std::uint8_t* out, std::size_t n) {
    const hn::ScalableTag<std::uint8_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto two = hn::Set(d, static_cast<std::uint8_t>(2));
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
        hn::StoreU(hn::Sub(two, hn::LoadU(d, a + i)), d, out + i);
    for (; i < n; ++i) out[i] = static_cast<std::uint8_t>(2 - a[i]);
}

}  // namespace HWY_NAMESPACE
}  // namespace qe::expr
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::expr {

HWY_EXPORT(AndImpl);
HWY_EXPORT(OrImpl);
HWY_EXPORT(NotImpl);

void logic_and_vec(const std::uint8_t* a, const std::uint8_t* b,
                   std::uint8_t* out, std::size_t n) {
    HWY_DYNAMIC_DISPATCH(AndImpl)(a, b, out, n);
}
void logic_or_vec(const std::uint8_t* a, const std::uint8_t* b,
                  std::uint8_t* out, std::size_t n) {
    HWY_DYNAMIC_DISPATCH(OrImpl)(a, b, out, n);
}
void logic_not_vec(const std::uint8_t* a, std::uint8_t* out, std::size_t n) {
    HWY_DYNAMIC_DISPATCH(NotImpl)(a, out, n);
}

}  // namespace qe::expr
#endif  // HWY_ONCE
