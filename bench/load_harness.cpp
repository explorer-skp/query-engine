// ported from raft-rsm/bench/load_gen.{h,cpp} — rigor infra, Raft-specifics stripped
#include "load_harness.h"

#include <chrono>
#include <thread>
#include <vector>

namespace qe::bench {

namespace {

inline std::chrono::steady_clock::time_point tpFromNs(std::uint64_t ns) {
    return std::chrono::steady_clock::time_point(
        std::chrono::nanoseconds(ns));
}

}  // namespace

LoadStats runClosedLoop(const WorkFn& work, const LoadConfig& cfg,
                        std::uint64_t measureStartNs,
                        std::uint64_t measureEndNs,
                        const std::atomic<bool>* stop) {
    std::vector<LoadStats> stats(static_cast<std::size_t>(cfg.threads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(cfg.threads));
    for (int t = 0; t < cfg.threads; ++t) {
        threads.emplace_back([&, t] {
            auto& local = stats[static_cast<std::size_t>(t)];
            XorShift64 rng(deriveSeed(cfg.seed, static_cast<std::uint64_t>(t)));
            for (;;) {
                if (stop != nullptr && stop->load(std::memory_order_relaxed)) {
                    break;
                }
                const std::uint64_t t0 = nowNs();
                if (stop == nullptr && t0 >= measureEndNs) break;
                const bool ok = work(rng, t);
                const std::uint64_t t1 = nowNs();
                // Only units fully inside the window count (standard
                // closed-loop windowing; partial units at the edges are
                // neither latency nor throughput samples).
                if (t0 >= measureStartNs && t1 < measureEndNs) {
                    ++local.scheduled;
                    if (ok) {
                        local.actual.record(t1 - t0);
                        ++local.oks;
                    } else {
                        ++local.failures;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    LoadStats merged;
    for (const auto& s : stats) merged.merge(s);
    return merged;
}

LoadStats runOpenLoop(const WorkFn& work, const LoadConfig& cfg,
                      std::uint64_t startNs, std::uint64_t measureStartNs,
                      std::uint64_t measureEndNs) {
    std::vector<LoadStats> stats(static_cast<std::size_t>(cfg.threads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(cfg.threads));
    // Total rate R striped across T threads: thread t fires at startNs + t/R,
    // then every T/R seconds — in aggregate one unit every 1/R seconds,
    // deterministically.
    const double perThreadIntervalNs =
        static_cast<double>(cfg.threads) * 1e9 / cfg.ratePerSec;
    const double offsetStepNs = 1e9 / cfg.ratePerSec;
    for (int t = 0; t < cfg.threads; ++t) {
        threads.emplace_back([&, t] {
            auto& local = stats[static_cast<std::size_t>(t)];
            XorShift64 rng(deriveSeed(cfg.seed, static_cast<std::uint64_t>(t)));
            const std::uint64_t offsetNs = static_cast<std::uint64_t>(
                static_cast<double>(t) * offsetStepNs);
            for (std::uint64_t k = 0;; ++k) {
                const std::uint64_t intended =
                    startNs + offsetNs +
                    static_cast<std::uint64_t>(static_cast<double>(k) *
                                               perThreadIntervalNs);
                if (intended >= measureEndNs) break;
                // The schedule is law: a late thread issues immediately
                // (sleep_until in the past returns at once) and the unit still
                // measures from `intended` — the CO correction.
                std::this_thread::sleep_until(tpFromNs(intended));
                const std::uint64_t send = nowNs();
                if (send > measureEndNs + cfg.drainCapNs) {
                    // Hopelessly oversaturated: count what the schedule still
                    // owed and mark the run (its tail is then a LOWER bound,
                    // never an overstatement).
                    const auto interval =
                        static_cast<std::uint64_t>(perThreadIntervalNs);
                    local.abandoned +=
                        (measureEndNs - intended) /
                            std::max<std::uint64_t>(interval, 1) +
                        1;
                    break;
                }
                const bool ok = work(rng, t);
                const std::uint64_t done = nowNs();
                if (intended >= measureStartNs) {
                    ++local.scheduled;
                    const std::uint64_t lateness =
                        send > intended ? send - intended : 0;
                    local.sumSendLatenessNs += lateness;
                    local.maxSendLatenessNs =
                        std::max(local.maxSendLatenessNs, lateness);
                    ++local.sends;
                    if (ok) {
                        local.intended.record(done - intended);
                        local.actual.record(done - send);
                        ++local.oks;
                    } else {
                        ++local.failures;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    LoadStats merged;
    for (const auto& s : stats) merged.merge(s);
    return merged;
}

}  // namespace qe::bench
