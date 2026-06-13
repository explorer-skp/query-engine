// WP-0: runtime Highway-target probe implementation. See tools/hwy_target.h.
//
// Performs ONE trivial dynamic-dispatch op so hwy::DispatchedTarget() reflects an
// actual call, then returns the chosen target's name. This is NOT an engine
// kernel — it exists only so the provenance JSON reports a real, detected target.

#include "tools/hwy_target.h"

#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tools/hwy_target.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe::tools {
namespace HWY_NAMESPACE {

// Inside the per-target namespace, HWY_TARGET is the concrete target of THIS
// compiled variant. Dynamic dispatch runs the variant for the target Highway
// chose on this CPU, so the returned name is exactly the dispatched target.
const char* CurrentTargetName() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace qe::tools
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe::tools {

HWY_EXPORT(CurrentTargetName);

std::string dispatchedHighwayTarget() {
    return HWY_DYNAMIC_DISPATCH(CurrentTargetName)();
}

}  // namespace qe::tools
#endif  // HWY_ONCE
