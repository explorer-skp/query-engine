// ported from raft-rsm/test/checker_selftest_test.cpp — rigor infra, Raft-specifics stripped
//
// A checker that never fails is worthless: each invariant checker over the
// harvested measurement infra is fed a CLEAN input it must accept and a
// deliberately-MUTATED input it must FLAG. This is the "a planted defect MUST
// be caught" discipline that every later work package's mutation self-test
// inherits — with all distributed-systems specifics removed. The checkers
// here guard the three things a benchmark number's credibility rests on: a
// monotone latency CDF, a working coordinated-omission correction, and an
// honest offered-rate count.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "histogram.h"

using qe::metrics::LatencyHistogram;

namespace {

constexpr std::uint64_t kMs = 1'000'000ull;

// --- Invariant checkers (each: empty == accepted) ---

// A quantile->value series is a CDF and must be non-decreasing in q.
std::vector<std::string> checkMonotoneQuantiles(
    const std::vector<std::pair<double, std::uint64_t>>& grid) {
    std::vector<std::string> v;
    for (std::size_t i = 1; i < grid.size(); ++i) {
        if (grid[i].second < grid[i - 1].second) {
            v.push_back("MONOTONE QUANTILE: value went backwards at q=" +
                        std::to_string(grid[i].first));
        }
    }
    return v;
}

// The coordinated-omission-corrected tail must (a) reach the floor a known
// injected stall implies and (b) exceed the uncorrected (actual-send) view. A
// harness that quietly measured from the actual send time violates both — the
// classic coordinated-omission defect.
std::vector<std::string> checkCoOmissionVisible(std::uint64_t correctedTailNs,
                                                std::uint64_t uncorrectedTailNs,
                                                std::uint64_t floorNs) {
    std::vector<std::string> v;
    if (correctedTailNs < floorNs) {
        v.push_back(
            "COORDINATED OMISSION: corrected tail below the injected-stall "
            "floor");
    }
    if (correctedTailNs < 2 * uncorrectedTailNs) {
        v.push_back(
            "COORDINATED OMISSION: corrected tail does not exceed the "
            "uncorrected view");
    }
    return v;
}

// The in-window scheduled count must match the offered schedule within edge
// tolerance; a count far below means the harness silently dropped late units.
std::vector<std::string> checkRateFidelity(std::uint64_t scheduled,
                                           std::uint64_t expected,
                                           std::uint64_t tol) {
    std::vector<std::string> v;
    const std::uint64_t lo = expected > tol ? expected - tol : 0;
    if (scheduled < lo || scheduled > expected + tol) {
        v.push_back("RATE FIDELITY: scheduled count outside tolerance");
    }
    return v;
}

std::vector<std::pair<double, std::uint64_t>> gridOf(const LatencyHistogram& h) {
    std::vector<std::pair<double, std::uint64_t>> g;
    for (const double q : {0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0}) {
        g.emplace_back(q, h.valueAtQuantile(q));
    }
    return g;
}

}  // namespace

TEST_CASE("monotone-quantile checker: accepts a real histogram grid, flags a "
          "planted inversion") {
    LatencyHistogram h;
    for (std::uint64_t v = 1; v <= 2000; ++v) h.record(v);
    auto grid = gridOf(h);  // real instrument output: monotone by construction
    CHECK(checkMonotoneQuantiles(grid).empty());

    // Planted defect: a mutated grid whose p99 dips below its p90 — exactly
    // what a broken valueAtQuantile would emit. The checker MUST catch it.
    auto mutated = grid;
    mutated[5].second = mutated[4].second / 2;  // q=0.99 pushed below q=0.9
    const auto v = checkMonotoneQuantiles(mutated);
    REQUIRE(v.size() == 1);
    CHECK(v[0].find("MONOTONE QUANTILE") != std::string::npos);
}

TEST_CASE("coordinated-omission checker: accepts a corrected tail, flags the "
          "omission defect") {
    // Two real histograms standing in for one run with a 480ms stall:
    //  corrected — every unit queued behind the stall recorded (CO-corrected),
    //  omitted   — only the handful in flight when the stall hit (the bug).
    LatencyHistogram corrected, omitted;
    for (int i = 0; i < 5000; ++i) {
        corrected.record(50'000);  // 50us steady-state
        omitted.record(50'000);
    }
    for (int i = 0; i < 700; ++i) corrected.record(480 * kMs);  // ~12% queued
    for (int i = 0; i < 6; ++i) omitted.record(480 * kMs);      // few in flight

    const auto correctedP99 = corrected.valueAtQuantile(0.99);
    const auto omittedP99 = omitted.valueAtQuantile(0.99);

    // Clean: the corrected tail reaches the floor and dwarfs the omitted view.
    CHECK(checkCoOmissionVisible(correctedP99, omittedP99, 250 * kMs).empty());

    // Planted defect: the harness reported the OMITTED (actual-send) tail as
    // its result — coordinated omission. The checker MUST flag it.
    const auto v = checkCoOmissionVisible(omittedP99, omittedP99, 250 * kMs);
    REQUIRE(v.size() == 2);
    CHECK(v[0].find("COORDINATED OMISSION") != std::string::npos);
}

TEST_CASE("rate-fidelity checker: accepts an on-rate count, flags silent "
          "drops") {
    // Clean: 3000 scheduled against a 3000 offered schedule, within edges.
    CHECK(checkRateFidelity(3000, 3000, 16).empty());

    // Planted defect: the harness silently skipped late units, so only 2400 of
    // 3000 were scheduled. The checker MUST flag the gap.
    const auto v = checkRateFidelity(2400, 3000, 16);
    REQUIRE(v.size() == 1);
    CHECK(v[0].find("RATE FIDELITY") != std::string::npos);
}
