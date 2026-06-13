// ported from raft-rsm/test/bench_selftest_test.cpp — rigor infra, Raft-specifics stripped
//
// The benchmark harness's own validation suite. A measurement instrument that
// cannot be shown to bite proves nothing, so each property here is
// demonstrated against the real harness driving a domain-free workload:
//
//  1. Workload reproducibility — same seed => identical draw sequence.
//  2. JSON writer sanity — the raw-data files must be well-formed.
//  3. Open-loop rate fidelity — the harness holds the target arrival rate
//     when the work keeps up.
//  4. Coordinated-omission stall self-test (the crux): inject an artificial
//     stall into the WORK mid-run and assert the intended-send-time tail
//     REPORTS it while the actual-send-time tail HIDES it. A harness that
//     hides an injected stall is broken — and this test is shown to fail if
//     the correction is removed.
//
// The injected "system" is a shared stall the work callable honors: when
// armed, every concurrent unit blocks until the stall ends, exactly as a
// real system-wide pause would queue arrivals behind it. No network, no RPC,
// no domain object is involved — only the rigor mechanism.
//
// Sanitizer builds run reduced rates/thresholds: ASan/TSan multiply CPU cost
// several-fold and this suite asserts wall-clock behavior; the sanitizer
// value here is race/UB coverage of the harness itself.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "bench_report.h"
#include "bench_util.h"
#include "doctest/doctest.h"
#include "load_harness.h"

using namespace qe::bench;

// Sanitizer detection across both toolchains: GCC defines __SANITIZE_*__,
// while clang/Apple-clang (our dev compiler) only exposes __has_feature. The
// source covered only the GCC spelling; this keeps the reduced-workload path
// active under the sanitizer gate on Apple Silicon too.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#    define QE_SANITIZED 1
#  endif
#endif
#if !defined(QE_SANITIZED) && \
    (defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__))
#  define QE_SANITIZED 1
#endif
#if !defined(QE_SANITIZED)
#  define QE_SANITIZED 0
#endif
constexpr bool kSanitized = QE_SANITIZED == 1;

namespace {

constexpr std::uint64_t kSecond = 1'000'000'000ull;

std::chrono::steady_clock::time_point tp(std::uint64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

// A domain-free stand-in for "the system under test": normally each unit of
// work is near-instant, but once armed every unit blocks until the stall
// deadline — modelling a system-wide pause that queues all in-flight and
// subsequent arrivals behind it. Shared across all harness threads.
struct StallableSystem {
    std::atomic<std::uint64_t> stallUntilNs{0};

    void arm(std::uint64_t ms) {
        stallUntilNs.store(nowNs() + ms * 1'000'000ull,
                           std::memory_order_relaxed);
    }

    // The injected WorkFn. Touches the seeded RNG (so the workload is
    // reproducible) and, if a stall is active, waits it out.
    bool serve(XorShift64& rng, int /*threadId*/) {
        (void)nextIndex(rng, 64);
        const std::uint64_t until = stallUntilNs.load(std::memory_order_relaxed);
        if (until > nowNs()) std::this_thread::sleep_until(tp(until));
        return true;
    }
};

}  // namespace

TEST_CASE("workload: same seed => same sequence, different seed => different") {
    XorShift64 a(deriveSeed(42, 7));
    XorShift64 b(deriveSeed(42, 7));
    XorShift64 c(deriveSeed(43, 7));
    bool anyDiff = false;
    for (int i = 0; i < 10000; ++i) {
        const auto ka = nextIndex(a, 64);
        REQUIRE(ka == nextIndex(b, 64));
        anyDiff = anyDiff || (ka != nextIndex(c, 64));
    }
    CHECK(anyDiff);
}

TEST_CASE("json writer: nesting, commas, escaping") {
    JsonWriter w;
    w.beginObject();
    w.kv("a", std::uint64_t{1});
    w.kv("b", "x\"y\\z\n");
    w.key("c");
    w.beginArray();
    w.value(std::uint64_t{1});
    w.beginObject();
    w.kv("d", true);
    w.endObject();
    w.endArray();
    w.kv("e", 2.5);
    w.endObject();
    CHECK(w.str() ==
          "{\"a\":1,\"b\":\"x\\\"y\\\\z\\n\",\"c\":[1,{\"d\":true}],"
          "\"e\":2.5}");
}

TEST_CASE("open-loop harness holds the target arrival rate") {
    const double rate = kSanitized ? 500.0 : 1000.0;
    StallableSystem sys;
    LoadConfig cfg;
    cfg.threads = 8;
    cfg.seed = 11;
    cfg.ratePerSec = rate;
    WorkFn work = [&](XorShift64& rng, int id) { return sys.serve(rng, id); };

    const std::uint64_t t0 = nowNs();
    const std::uint64_t ms = t0 + 1 * kSecond;   // 1s warmup
    const std::uint64_t me = ms + 3 * kSecond;   // 3s window
    const auto stats = runOpenLoop(work, cfg, t0, ms, me);

    const auto expected = static_cast<std::uint64_t>(rate * 3);
    CAPTURE(stats.scheduled);
    CAPTURE(stats.sends);
    CAPTURE(stats.failures);
    CHECK(stats.failures == 0);
    CHECK(stats.abandoned == 0);
    // The schedule covers the window to within edge effects (one unit per
    // thread at each boundary).
    CHECK(stats.scheduled >= expected - 16);
    CHECK(stats.scheduled <= expected + 16);
    // Sends track the schedule: when the work keeps up, dispatch lateness
    // stays well below the request interval (8ms / thread here).
    const double meanLateNs =
        static_cast<double>(stats.sumSendLatenessNs) /
        static_cast<double>(stats.sends ? stats.sends : 1);
    CAPTURE(meanLateNs);
    CHECK(meanLateNs < (kSanitized ? 30e6 : 4e6));
}

TEST_CASE("coordinated-omission stall self-test: the reported tail must move") {
    // Open-loop run with a 500ms stall injected ~1.5s into a 4s measurement
    // window. ~12% of the window's units are scheduled during the stall:
    // their queueing is real and MUST appear in the intended-send-time tail.
    const double rate = kSanitized ? 600.0 : 1500.0;
    StallableSystem sys;
    LoadConfig cfg;
    cfg.threads = 6;
    cfg.seed = 12;
    cfg.ratePerSec = rate;
    WorkFn work = [&](XorShift64& rng, int id) { return sys.serve(rng, id); };

    const std::uint64_t t0 = nowNs();
    const std::uint64_t ms = t0 + 1 * kSecond;
    const std::uint64_t me = ms + 4 * kSecond;

    std::thread arm([&] {
        std::this_thread::sleep_until(tp(ms + 1500 * 1000 * 1000ull));
        sys.arm(/*ms=*/500);
    });
    const auto stats = runOpenLoop(work, cfg, t0, ms, me);
    arm.join();

    const auto intendedP99 = stats.intended.valueAtQuantile(0.99);
    const auto intendedP999 = stats.intended.valueAtQuantile(0.999);
    const auto actualP99 = stats.actual.valueAtQuantile(0.99);
    CAPTURE(intendedP99);
    CAPTURE(intendedP999);
    CAPTURE(actualP99);
    CAPTURE(stats.oks);
    CHECK(stats.failures == 0);

    // The corrected tail reports the stall (>= ~250ms of queueing at p99 for
    // a 500ms stall over ~12% of samples; generous margin below the
    // theoretical ~490ms).
    CHECK(intendedP99 >= 250 * 1000 * 1000ull);
    CHECK(intendedP999 >= 350 * 1000 * 1000ull);
    // The uncorrected (actual-send-time) view hides most of it: only the units
    // already in flight when the stall hit see it; everything queued behind
    // them dispatches late and measures small. This gap IS coordinated
    // omission, demonstrated. Removing the correction (recording done-send in
    // place of done-intended) collapses this gap and fails the assertion.
    CHECK(intendedP99 >= 2 * actualP99);
    if (!kSanitized) {
        CHECK(actualP99 <= 150 * 1000 * 1000ull);
    }
    std::printf(
        "stall self-test: intended p99=%.1fms p99.9=%.1fms | actual "
        "p99=%.1fms (the correction made the stall visible)\n",
        static_cast<double>(intendedP99) / 1e6,
        static_cast<double>(intendedP999) / 1e6,
        static_cast<double>(actualP99) / 1e6);
}
