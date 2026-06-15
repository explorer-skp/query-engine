//  WP-10: COORDINATED-OMISSION-correct latency for a REAL engine query under
//  SUSTAINED load. The harvested open-loop load harness (bench/load_harness.h)
//  issues queries at a fixed target rate and measures each one FROM ITS INTENDED
//  SEND TIME — so when the engine (or the machine) stalls, the units queued behind
//  the stall record the queueing they actually suffered instead of being silently
//  rescheduled. The actual-send-time histogram is kept alongside to SHOW the
//  correction working: inject a stall and the intended tail climbs while the
//  actual tail stays flat. That gap IS coordinated omission, demonstrated on the
//  engine itself rather than a synthetic stand-in (the synthetic mechanism proof
//  lives in tests/coord_omission_selftest_test.cpp; this WIRES it to a query).
//
//  The WorkFn lowers a FRESH operator tree from the shared immutable Plan each
//  call (Scan only ever reads the borrowed Table), so the T harness threads run
//  concurrent read-only queries with no shared mutable operator state.
//
//  HONESTY (§2): Mac results are preliminary/relative-only; the CO CORRECTION and
//  its self-test are host-independent rigor (they assert a RATIO between two views
//  of the same run), so they are credible on Mac. Determinism: seed printed.
//  Usage: bench_sustained [--n ROWS] [--threads T] [--rate R] [--warmup-s W]
//         [--window-s S] [--inject-stall-ms MS] [--stall-at-s T] [--seed N]
//         [--host TAG] [--isa TAG] [--outdir DIR]

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "bench_harness.h"
#include "bench_report.h"
#include "bench_util.h"
#include "load_harness.h"
#include "validity_gate.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/operator.h"
#include "ops/table.h"
#include "plan/plan.h"
#include "tools/hwy_target.h"

using namespace qe;
using namespace qe::bench;

namespace {

std::string argValue(int argc, char** argv, std::string_view flag,
                     std::string def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}

std::string deriveHostTag(const std::string& brand) {
#if defined(__APPLE__)
    const auto pos = brand.find('M');
    for (std::size_t i = pos; i != std::string::npos && i + 1 < brand.size();
         i = brand.find('M', i + 1))
        if (brand[i + 1] >= '0' && brand[i + 1] <= '9') {
            std::string gen = "m";
            std::size_t j = i + 1;
            while (j < brand.size() && brand[j] >= '0' && brand[j] <= '9')
                gen += brand[j++];
            return "mac-" + gen;
        }
    return "mac";
#else
    (void)brand;
    return "linux-host";
#endif
}

std::chrono::steady_clock::time_point tpNs(std::uint64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

Table makeTable(std::size_t n, std::uint64_t seed) {
    Schema s;
    s.fields.emplace_back("k", Type::I64);
    s.fields.emplace_back("v", Type::I64);
    XorShift64 rng(seed ^ 0x5057Au);
    std::vector<OwnedColumn> cols;
    OwnedColumn kc = OwnedColumn::make(Type::I64, n);
    OwnedColumn vc = OwnedColumn::make(Type::I64, n);
    auto* kd = reinterpret_cast<std::int64_t*>(kc.mutable_data());
    auto* vd = reinterpret_cast<std::int64_t*>(vc.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        kd[i] = static_cast<std::int64_t>(rng.next() % 32);
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

std::uint64_t drainChecksum(Operator& op) {
    op.open();
    std::uint64_t cs = 0;
    while (auto b = op.next()) cs += b->row_count;
    op.close();
    return cs;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = static_cast<std::size_t>(std::strtoull(
        argValue(argc, argv, "--n", "4096").c_str(), nullptr, 10));
    const int threads = static_cast<int>(
        std::strtol(argValue(argc, argv, "--threads", "6").c_str(), nullptr, 10));
    const double rate =
        std::strtod(argValue(argc, argv, "--rate", "2000").c_str(), nullptr);
    const double warmupS =
        std::strtod(argValue(argc, argv, "--warmup-s", "1").c_str(), nullptr);
    const double windowS =
        std::strtod(argValue(argc, argv, "--window-s", "4").c_str(), nullptr);
    const std::uint64_t stallMs = std::strtoull(
        argValue(argc, argv, "--inject-stall-ms", "0").c_str(), nullptr, 10);
    const double stallAtS =
        std::strtod(argValue(argc, argv, "--stall-at-s", "1.5").c_str(), nullptr);
    const std::uint64_t seed = std::strtoull(
        argValue(argc, argv, "--seed", "20260615").c_str(), nullptr, 10);
    const std::string outdir = argValue(argc, argv, "--outdir", "");
    const std::string label =
        stallMs ? "engine_sustained_stall" : "engine_sustained";

    MachineState machine = captureMachineState();
    machine.hwyTarget = qe::tools::dispatchedHighwayTarget();
    std::string host = argValue(argc, argv, "--host", "");
    if (host.empty()) host = deriveHostTag(machine.cpuModel);
    std::string isa = argValue(argc, argv, "--isa", "");
    if (isa.empty()) isa = machine.isa;
#if defined(__APPLE__)
    const bool preliminary = true;
#else
    const bool preliminary = false;
#endif

    std::fprintf(
        stderr,
        "seed=%llu (replay: bench_sustained --n %zu --threads %d --rate %.0f "
        "--window-s %.0f --inject-stall-ms %llu --seed %llu)\n",
        static_cast<unsigned long long>(seed), n, threads, rate, windowS,
        static_cast<unsigned long long>(stallMs),
        static_cast<unsigned long long>(seed));

    // Gate informs validity but does NOT gate the CO self-property: the CO
    // correction is a ratio internal to the run, credible even on a busy host.
    const GateVerdict gate = evaluateGate(machine);

    Table t = makeTable(n, seed);
    const qe::plan::Plan plan = makePlan(t);

    // Shared injected stall (a system-wide pause queueing all in-flight work).
    std::atomic<std::uint64_t> stallUntilNs{0};
    WorkFn work = [&](XorShift64& rng, int /*tid*/) {
        (void)rng.next();
        auto tree = plan.lower();
        volatile std::uint64_t sink = drainChecksum(*tree);
        (void)sink;
        const std::uint64_t until =
            stallUntilNs.load(std::memory_order_relaxed);
        if (until > nowNs()) std::this_thread::sleep_until(tpNs(until));
        return true;
    };

    LoadConfig cfg;
    cfg.threads = threads;
    cfg.seed = seed;
    cfg.ratePerSec = rate;

    const std::uint64_t t0 = nowNs();
    const std::uint64_t ms =
        t0 + static_cast<std::uint64_t>(warmupS * 1e9);
    const std::uint64_t me =
        ms + static_cast<std::uint64_t>(windowS * 1e9);

    std::thread armer;
    if (stallMs) {
        armer = std::thread([&] {
            std::this_thread::sleep_until(
                tpNs(ms + static_cast<std::uint64_t>(stallAtS * 1e9)));
            stallUntilNs.store(nowNs() + stallMs * 1'000'000ull,
                               std::memory_order_relaxed);
        });
    }
    const LoadStats stats = runOpenLoop(work, cfg, t0, ms, me);
    if (armer.joinable()) armer.join();

    const auto iP99 = stats.intended.valueAtQuantile(0.99);
    const auto aP99 = stats.actual.valueAtQuantile(0.99);
    const double coRatio =
        aP99 > 0 ? static_cast<double>(iP99) / static_cast<double>(aP99) : 0.0;
    const double thrCps =
        windowS > 0 ? static_cast<double>(stats.oks) / windowS : 0.0;

    JsonWriter w;
    w.beginObject();
    writeRunTags(
        w, "bench_sustained", label, host, isa, seed, preliminary,
        "coordinated-omission-correct engine latency under sustained open-loop "
        "load; intended-vs-actual send time. CO correction is host-independent.");
    w.key("config");
    w.beginObject();
    w.kv("rows", static_cast<std::uint64_t>(n));
    w.kv("threads", threads);
    w.kv("rate_per_sec", rate);
    w.kv("warmup_s", warmupS);
    w.kv("window_s", windowS);
    w.kv("injected_stall_ms", stallMs);
    w.kv("query", plan.to_string());
    w.endObject();
    w.key("machine_start");
    writeMachine(w, machine);
    w.key("validity");
    writeGateVerdict(w, gate);
    w.key("results");
    w.beginObject();
    w.kv("valid", gate.accepted);
    w.kv("throughput_cps", thrCps);
    w.kv("scheduled", stats.scheduled);
    w.kv("oks", stats.oks);
    w.kv("failures", stats.failures);
    w.kv("abandoned", stats.abandoned);
    w.kv("max_send_lateness_ns", stats.maxSendLatenessNs);
    w.kv("injected_stall_ms", stallMs);
    w.kv("co_ratio_p99_intended_over_actual", coRatio);
    w.key("e2e_intended");
    writeHistogram(w, stats.intended);
    w.key("e2e_actual");
    writeHistogram(w, stats.actual);
    w.endObject();
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    std::fprintf(stderr,
                 "sustained: thr=%.0f q/s | intended p99=%.2fms actual "
                 "p99=%.2fms | CO ratio=%.1fx%s\n",
                 thrCps, static_cast<double>(iP99) / 1e6,
                 static_cast<double>(aP99) / 1e6, coRatio,
                 stallMs ? " (stall injected)" : "");

    if (!outdir.empty()) {
        const std::string path =
            outdir + "/" + label + "_" + host + "_" + isa + ".json";
        if (!writeFile(path, w.str() + "\n"))
            std::fprintf(stderr, "failed to write %s\n", path.c_str());
    }
    return 0;
}
