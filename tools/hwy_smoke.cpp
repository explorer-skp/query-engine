// WP-0: Google Highway integration smoke test.
//
// Proves Highway is build-available and that runtime dispatch works on this host.
// It is NOT an engine kernel (no engine behavior ships in WP-0): it runs ONE
// trivial vectorized op (sum a float array) through Highway's dynamic dispatch,
// then prints the target Highway actually CHOSE at runtime.
//
// The chosen target is detected, never hardcoded — that is the whole point of
// Highway and of the project's "portable SIMD, no hardcoded width/ISA" rule.
//
// Usage: hwy_smoke [--target-only]
//   --target-only  print just the dispatched target string (used to wire the
//                  real value into the host-state provenance JSON; see
//                  scripts/host_state.sh and bench/host_state_main.cpp).

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

// Highway dynamic-dispatch boilerplate: foreach_target.h re-includes THIS file
// once per supported target. The path is resolved from the repo-root include dir.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tools/hwy_smoke.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace qe_smoke {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Trivial vectorized reduction — the one op we use to exercise dispatch.
float SumImpl(const float* x, size_t n) {
    const hn::ScalableTag<float> d;
    auto acc = hn::Zero(d);
    size_t i = 0;
    const size_t lanes = hn::Lanes(d);
    for (; i + lanes <= n; i += lanes) {
        acc = hn::Add(acc, hn::LoadU(d, x + i));
    }
    float total = hn::ReduceSum(d, acc);
    for (; i < n; ++i) total += x[i];  // scalar tail
    return total;
}

// HWY_TARGET inside this namespace is the concrete target of the compiled
// variant; dispatched, it names the target Highway actually selected.
const char* CurrentTargetName() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace qe_smoke
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace qe_smoke {

HWY_EXPORT(SumImpl);
HWY_EXPORT(CurrentTargetName);

float Sum(const float* x, size_t n) {
    return HWY_DYNAMIC_DISPATCH(SumImpl)(x, n);
}
const char* DispatchedTargetName() {
    return HWY_DYNAMIC_DISPATCH(CurrentTargetName)();
}

}  // namespace qe_smoke

int main(int argc, char** argv) {
    const bool target_only =
        (argc > 1) && std::strcmp(argv[1], "--target-only") == 0;

    // One real dispatched op so the chosen target reflects an actual call.
    std::vector<float> data(10000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = 1.0f;
    const float sum = qe_smoke::Sum(data.data(), data.size());

    // The target Highway chose for this machine, at runtime.
    const char* chosen = qe_smoke::DispatchedTargetName();

    if (target_only) {
        std::printf("%s\n", chosen);
        return 0;
    }

    std::printf("highway_dispatched_target=%s\n", chosen);
    std::printf("highway_static_target=%s\n", hwy::TargetName(HWY_STATIC_TARGET));
    std::printf("highway_version=%d.%d.%d\n", HWY_MAJOR, HWY_MINOR, HWY_PATCH);
    std::printf("smoke_sum=%.1f (expected %.1f)\n", sum,
                static_cast<double>(data.size()));
    // Non-zero exit if the trivial op miscomputed — a dispatch that runs the
    // wrong/no kernel must not pass silently.
    return (sum == static_cast<float>(data.size())) ? 0 : 1;
}
#endif  // HWY_ONCE
