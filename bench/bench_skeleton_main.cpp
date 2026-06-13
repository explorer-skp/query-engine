// ported from raft-rsm/bench (rsm_bench driver) — rigor infra, Raft-specifics stripped
//
// A minimal end-to-end exercise of the harvested harness: it captures the host
// record, drives a domain-free open-loop workload through the
// coordinated-omission load harness, and emits the standard per-run JSON the
// plots and tables regenerate from (machine state + intended/actual histograms
// + counters). The "work" here is a near-instant placeholder; a real work
// package swaps in its operator and keeps every line of the reporting path.
//
// Determinism: the seed is printed and the run replays from a single command.
//
// Usage: bench_skeleton [--seed N] [--rate R] [--threads T] [--window-ms W]
//                       [--warmup-ms M] [--stall-ms S] [--host TAG] [--isa TAG]
//                       [--out PATH] [--label L]

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>

#include "bench_report.h"
#include "bench_util.h"
#include "load_harness.h"
#include "tools/hwy_target.h"

using namespace qe::bench;

namespace {

std::string argValue(int argc, char** argv, std::string_view flag,
                     std::string def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return def;
}

std::chrono::steady_clock::time_point tp(std::uint64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

std::string deriveHostTag(const std::string& brand) {
#if defined(__APPLE__)
    return brand.find("Apple") != std::string::npos ? "mac" : "mac";
#else
    (void)brand;
    return "linux-host";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t seed =
        std::strtoull(argValue(argc, argv, "--seed", "1").c_str(), nullptr, 10);
    const double rate = std::strtod(
        argValue(argc, argv, "--rate", "2000").c_str(), nullptr);
    const int threads = static_cast<int>(std::strtol(
        argValue(argc, argv, "--threads", "4").c_str(), nullptr, 10));
    const std::uint64_t warmupMs = std::strtoull(
        argValue(argc, argv, "--warmup-ms", "300").c_str(), nullptr, 10);
    const std::uint64_t windowMs = std::strtoull(
        argValue(argc, argv, "--window-ms", "700").c_str(), nullptr, 10);
    const std::uint64_t stallMs = std::strtoull(
        argValue(argc, argv, "--stall-ms", "0").c_str(), nullptr, 10);
    const std::string label = argValue(argc, argv, "--label", "co_demo");
    const std::string outPath = argValue(argc, argv, "--out", "");

    MachineState machine = captureMachineState();
    // Real runtime Highway target (detected, not the compile-time fallback).
    machine.hwyTarget = qe::tools::dispatchedHighwayTarget();
    std::string host = argValue(argc, argv, "--host", "");
    if (host.empty()) host = deriveHostTag(machine.cpuModel);
    std::string isa = argValue(argc, argv, "--isa", "");
    if (isa.empty()) isa = machine.isa;

    // Replay banner: a single command reproduces this run.
    std::fprintf(stderr,
                 "seed=%llu (replay: bench_skeleton --seed %llu --rate %.0f "
                 "--threads %d --warmup-ms %llu --window-ms %llu --stall-ms "
                 "%llu)\n",
                 static_cast<unsigned long long>(seed),
                 static_cast<unsigned long long>(seed), rate, threads,
                 static_cast<unsigned long long>(warmupMs),
                 static_cast<unsigned long long>(windowMs),
                 static_cast<unsigned long long>(stallMs));

    // A shared, near-instant "system" with an optional one-shot stall — the
    // same injectable mechanism the CO self-test uses, here behind a flag.
    std::atomic<std::uint64_t> stallUntilNs{0};
    WorkFn work = [&](XorShift64& rng, int) {
        (void)nextIndex(rng, 64);
        const std::uint64_t until = stallUntilNs.load(std::memory_order_relaxed);
        if (until > nowNs()) std::this_thread::sleep_until(tp(until));
        return true;
    };

    LoadConfig cfg;
    cfg.threads = threads;
    cfg.seed = seed;
    cfg.ratePerSec = rate;

    const std::uint64_t t0 = nowNs();
    const std::uint64_t ms = t0 + warmupMs * 1'000'000ull;
    const std::uint64_t me = ms + windowMs * 1'000'000ull;

    std::thread arm;
    if (stallMs > 0) {
        arm = std::thread([&] {
            std::this_thread::sleep_until(tp(ms + (windowMs / 2) * 1'000'000ull));
            stallUntilNs.store(nowNs() + stallMs * 1'000'000ull,
                               std::memory_order_relaxed);
        });
    }
    const auto stats = runOpenLoop(work, cfg, t0, ms, me);
    if (arm.joinable()) arm.join();

    const double windowSec = static_cast<double>(windowMs) / 1000.0;
    const double throughput =
        windowSec > 0 ? static_cast<double>(stats.oks) / windowSec : 0.0;
    const double meanLateNs =
        static_cast<double>(stats.sumSendLatenessNs) /
        static_cast<double>(stats.sends ? stats.sends : 1);
    const bool valid = stats.failures == 0 && stats.abandoned == 0;

    JsonWriter w;
    w.beginObject();
    w.kv("kind", "bench_skeleton");
    w.kv("label", label);
    w.kv("host", host);
    w.kv("isa", isa);
    w.kv("seed", seed);
    w.key("config");
    w.beginObject();
    w.kv("threads", threads);
    w.kv("rate", rate);
    w.kv("warmup_ms", warmupMs);
    w.kv("window_ms", windowMs);
    w.kv("stall_ms", stallMs);
    w.endObject();
    w.key("machine_start");
    writeMachine(w, machine);
    w.key("results");
    w.beginObject();
    w.kv("scheduled", stats.scheduled);
    w.kv("oks", stats.oks);
    w.kv("failures", stats.failures);
    w.kv("abandoned", stats.abandoned);
    w.kv("throughput_cps", throughput);
    w.kv("mean_send_lateness_ns", meanLateNs);
    w.kv("max_send_lateness_ns", stats.maxSendLatenessNs);
    w.kv("valid", valid);
    w.key("e2e_intended");
    writeHistogram(w, stats.intended);
    w.key("e2e_actual");
    writeHistogram(w, stats.actual);
    w.endObject();
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    if (!outPath.empty()) {
        if (!writeFile(outPath, w.str() + "\n")) {
            std::fprintf(stderr, "failed to write %s\n", outPath.c_str());
            return 1;
        }
        std::fprintf(stderr, "wrote %s\n", outPath.c_str());
    }
    return 0;
}
