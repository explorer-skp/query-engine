//  WP-7 benchmark: the permutation-GATHER vec-vs-scalar RELATIVE ratio — the first
//  vec-vs-scalar number in the engine. Whole-row movement in a sort is a gather by
//  the sorted row permutation (simd/gather_kernels.h); this driver times the
//  Highway gather64 kernel against its independently-written scalar twin on the
//  SAME scattered-index workload, and reports p50/p99 of each plus the speedup
//  ratio, as JSON tagged host+isa.
//
//  HOST HONESTY (RIGOR.md §2). Mac numbers are PRELIMINARY / RELATIVE-ONLY: the
//  emitted JSON carries `"preliminary": true` and no absolute/AVX-512/roofline
//  claim. The headline absolute number, if ever wanted, comes from the x86 box.
//
//  Determinism: the seed is printed and the run replays from a single command.
//  Usage: bench_sort [--n N] [--iters I] [--seed S] [--host TAG] [--isa TAG]
//                    [--out PATH] [--label L]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "bench_report.h"
#include "bench_util.h"
#include "histogram.h"
#include "validity_gate.h"
#include "simd/gather_kernels.h"
#include "tools/hwy_target.h"

using namespace qe::bench;
using qe::metrics::LatencyHistogram;

namespace {

std::string argValue(int argc, char** argv, std::string_view flag,
                     std::string def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}

std::string deriveHostTag(const std::string& brand) {
#if defined(__APPLE__)
    (void)brand;
    return "mac";
#else
    (void)brand;
    return "linux-host";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = static_cast<std::size_t>(std::strtoull(
        argValue(argc, argv, "--n", "1048576").c_str(), nullptr, 10));
    const int iters = static_cast<int>(std::strtol(
        argValue(argc, argv, "--iters", "200").c_str(), nullptr, 10));
    const std::uint64_t seed = std::strtoull(
        argValue(argc, argv, "--seed", "20260615").c_str(), nullptr, 10);
    const std::string label = argValue(argc, argv, "--label", "gather64");
    const std::string outPath = argValue(argc, argv, "--out", "");

    MachineState machine = captureMachineState();
    machine.hwyTarget = qe::tools::dispatchedHighwayTarget();
    // VALIDITY GATE (audit H9): this driver emits host+isa-tagged ratio JSON,
    // so a contaminated run must be rejected, not reported — the same rule
    // bench_ops/engine_vs_duckdb already enforce ("gate-checks before report").
    const GateVerdict gate = evaluateGate(machine);
    if (!gate.accepted) {
        std::fprintf(stderr,
                     "VALIDITY GATE: REJECTED — run is contaminated, numbers are "
                     "NOT credible:\n");
        for (const auto& r : gate.reasons)
            std::fprintf(stderr, "    - %s\n", r.c_str());
    } else {
        std::fprintf(stderr,
                     "VALIDITY GATE: accepted (loadavg/cpu=%.2f probe_spread=%.2f)\n",
                     gate.loadavg_per_cpu, gate.probe_spread);
    }
    std::string host = argValue(argc, argv, "--host", "");
    if (host.empty()) host = deriveHostTag(machine.cpuModel);
    std::string isa = argValue(argc, argv, "--isa", "");
    if (isa.empty()) isa = machine.isa;

    std::fprintf(stderr,
                 "seed=%llu (replay: bench_sort --n %zu --iters %d --seed %llu)\n",
                 static_cast<unsigned long long>(seed), n, iters,
                 static_cast<unsigned long long>(seed));

    // Source values + a SCATTERED index stream (models gathering rows by a sorted
    // permutation: reads jump around the source). One XorShift stream, seeded.
    XorShift64 rng(seed ? seed : 1);
    std::vector<std::uint64_t> src(n);
    std::vector<std::uint32_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = rng.next();
    for (std::size_t i = 0; i < n; ++i)
        idx[i] = static_cast<std::uint32_t>(rng.next() % n);
    std::vector<std::uint64_t> out(n);

    // Warmup (page-in + branch/predictor warmup), excluded from the histograms.
    for (int w = 0; w < 3; ++w) {
        qe::simd::gather64_vec(src.data(), idx.data(), n, out.data());
        qe::simd::gather64_scalar(src.data(), idx.data(), n, out.data());
    }

    LatencyHistogram hv, hs;
    std::uint64_t checksum = 0;  // defeat dead-store elimination
    for (int it = 0; it < iters; ++it) {
        std::uint64_t t0 = nowNs();
        qe::simd::gather64_vec(src.data(), idx.data(), n, out.data());
        std::uint64_t t1 = nowNs();
        hv.record(t1 - t0);
        checksum ^= out[(static_cast<std::size_t>(it) * 2654435761u) % n];

        t0 = nowNs();
        qe::simd::gather64_scalar(src.data(), idx.data(), n, out.data());
        t1 = nowNs();
        hs.record(t1 - t0);
        checksum ^= out[(static_cast<std::size_t>(it) * 40503u) % n];
    }

    const double vec_p50 = static_cast<double>(hv.valueAtQuantile(0.50));
    const double vec_p99 = static_cast<double>(hv.valueAtQuantile(0.99));
    const double sca_p50 = static_cast<double>(hs.valueAtQuantile(0.50));
    const double sca_p99 = static_cast<double>(hs.valueAtQuantile(0.99));
    const double speedup_p50 = vec_p50 > 0 ? sca_p50 / vec_p50 : 0.0;
    const double speedup_p99 = vec_p99 > 0 ? sca_p99 / vec_p99 : 0.0;

    JsonWriter w;
    w.beginObject();
    w.kv("kind", "bench_sort");
    w.kv("label", label);
    w.kv("host", host);
    w.kv("isa", isa);
    w.kv("gate_accepted", gate.accepted ? "true" : "false");
    w.kv("seed", seed);
    w.kv("preliminary", true);  // Mac: relative-only; no absolute/roofline claim
    w.kv("note",
         "vec-vs-scalar permutation gather; relative ratio only (RIGOR.md S2)");
    w.key("config");
    w.beginObject();
    w.kv("elements", static_cast<std::uint64_t>(n));
    w.kv("bytes_per_element", static_cast<std::uint64_t>(8));
    w.kv("iters", iters);
    w.kv("checksum", checksum);
    w.endObject();
    w.key("machine_start");
    writeMachine(w, machine);
    w.key("results");
    w.beginObject();
    w.key("gather64_vec_ns");
    writeHistogram(w, hv);
    w.key("gather64_scalar_ns");
    writeHistogram(w, hs);
    w.kv("speedup_p50_scalar_over_vec", speedup_p50);
    w.kv("speedup_p99_scalar_over_vec", speedup_p99);
    w.endObject();
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    std::fprintf(stderr,
                 "vec p50=%.0fns p99=%.0fns | scalar p50=%.0fns p99=%.0fns | "
                 "speedup(scalar/vec) p50=%.2fx p99=%.2fx\n",
                 vec_p50, vec_p99, sca_p50, sca_p99, speedup_p50, speedup_p99);
    if (!outPath.empty() && !writeFile(outPath, w.str() + "\n")) {
        std::fprintf(stderr, "failed to write %s\n", outPath.c_str());
        return 1;
    }
    return gate.accepted ? 0 : 3;
}
