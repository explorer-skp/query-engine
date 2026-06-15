//  WP-10: the coordinated-omission correction, demonstrated on the REAL ENGINE.
//  The harvested self-test (tests/coord_omission_selftest_test.cpp) proves the
//  mechanism against a synthetic stallable stand-in; this WIRES it to an actual
//  query: the open-loop harness drives a lowered Plan at a fixed rate, a stall is
//  injected mid-window, and we assert the INTENDED-send-time tail REPORTS the
//  stall while the ACTUAL-send-time tail HIDES it. Remove the correction (record
//  done-send instead of done-intended in load_harness.cpp) and the gap collapses
//  and this fails — so the correction cannot silently regress.
//
//  Each WorkFn call lowers its OWN operator tree from the shared immutable Plan
//  (Scan only reads the borrowed Table), so the harness threads run concurrent
//  read-only queries with no shared mutable operator state.
//
//  Host-independent: the assertions are RATIOS between two views of ONE run, so
//  they hold on Mac (preliminary for absolute timing, but the CO property is not
//  an absolute number). Replay: ./co_engine_selftest_test --seed N

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "bench_util.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "load_harness.h"
#include "ops/aggregate.h"
#include "ops/operator.h"
#include "ops/table.h"
#include "plan/plan.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::bench;

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

Table makeTable(std::size_t n, std::uint64_t seed) {
    Schema s;
    s.fields.emplace_back("k", Type::I64);
    s.fields.emplace_back("v", Type::I64);
    XorShift64 rng(seed ? seed : 1);
    std::vector<OwnedColumn> cols;
    OwnedColumn kc = OwnedColumn::make(Type::I64, n);
    OwnedColumn vc = OwnedColumn::make(Type::I64, n);
    auto* kd = reinterpret_cast<std::int64_t*>(kc.mutable_data());
    auto* vd = reinterpret_cast<std::int64_t*>(vc.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        kd[i] = static_cast<std::int64_t>(rng.next() % 16);
        vd[i] = static_cast<std::int64_t>(rng.next() % 2000) - 1000;
    }
    cols.push_back(std::move(kc));
    cols.push_back(std::move(vc));
    return Table(s, std::move(cols));
}

qe::plan::Plan makePlan(const Table& t) {
    using namespace qe::expr;
    using namespace qe::plan;
    return scan(t)
        .filter(gt(col(Type::I64, 1), lit(Scalar::i64(0))))
        .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(1, "tot")})
        .plan();
}

std::uint64_t drain(Operator& op) {
    op.open();
    std::uint64_t cs = 0;
    while (auto b = op.next()) cs += b->row_count;
    op.close();
    return cs;
}

}  // namespace

TEST_CASE("CO on the engine: injected stall moves the intended tail, not actual") {
    const std::uint64_t seed = qe::test::seed();
    const double rate = kSanitized ? 400.0 : 1200.0;
    const std::size_t rows = kSanitized ? 1024 : 2048;

    Table t = makeTable(rows, seed);
    const qe::plan::Plan plan = makePlan(t);

    std::atomic<std::uint64_t> stallUntilNs{0};
    WorkFn work = [&](XorShift64& rng, int /*tid*/) {
        (void)rng.next();
        auto tree = plan.lower();
        volatile std::uint64_t sink = drain(*tree);
        (void)sink;
        const std::uint64_t until = stallUntilNs.load(std::memory_order_relaxed);
        if (until > nowNs()) std::this_thread::sleep_until(tp(until));
        return true;
    };

    LoadConfig cfg;
    cfg.threads = 6;
    cfg.seed = seed;
    cfg.ratePerSec = rate;

    const std::uint64_t t0 = nowNs();
    const std::uint64_t ms = t0 + 1 * kSecond;
    const std::uint64_t me = ms + 4 * kSecond;

    std::thread arm([&] {
        std::this_thread::sleep_until(tp(ms + 1500 * 1000 * 1000ull));
        stallUntilNs.store(nowNs() + 500 * 1'000'000ull,  // 500ms system pause
                           std::memory_order_relaxed);
    });
    const LoadStats stats = runOpenLoop(work, cfg, t0, ms, me);
    arm.join();

    const auto intendedP99 = stats.intended.valueAtQuantile(0.99);
    const auto intendedP999 = stats.intended.valueAtQuantile(0.999);
    const auto actualP99 = stats.actual.valueAtQuantile(0.99);
    CAPTURE(intendedP99);
    CAPTURE(intendedP999);
    CAPTURE(actualP99);
    CAPTURE(stats.oks);
    CHECK(stats.failures == 0);

    // The corrected (intended-send-time) tail reports the 500ms pause; queued
    // units see >= ~250ms at p99 (generous margin below the ~490ms theoretical).
    CHECK(intendedP99 >= 250 * 1000 * 1000ull);
    CHECK(intendedP999 >= 350 * 1000 * 1000ull);
    // The uncorrected (actual-send-time) tail hides most of it: this gap IS
    // coordinated omission, demonstrated on the engine workload.
    CHECK(intendedP99 >= 2 * actualP99);
    std::printf(
        "CO-on-engine: intended p99=%.1fms p99.9=%.1fms | actual p99=%.1fms "
        "(correction made the stall visible)\n",
        static_cast<double>(intendedP99) / 1e6,
        static_cast<double>(intendedP999) / 1e6,
        static_cast<double>(actualP99) / 1e6);
}
