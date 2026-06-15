//  WP-10: the ENGINE-VS-DUCKDB relative ratio. The same query — assembled ONCE
//  through the frozen plan builder (plan/plan.h) — is run through this engine's
//  operator tree and through DuckDB (oracle/duckdb_oracle.h, the authoritative
//  golden model, compiled only when the amalgamation is staged, QE_WITH_DUCKDB),
//  and their timings are ratioed. One description, both backends — the same single
//  source of truth the differential oracle uses, so the thing being timed on both
//  sides is provably the same query.
//
//  HONEST ASYMMETRY (stated, not hidden — §2): the engine times EXECUTION over a
//  table already resident in its columnar format; the DuckDB side goes through
//  run_plan_duckdb(), which (re)loads the base table into an in-memory DuckDB per
//  call, so its time includes ingest. This is the only seam the oracle exposes and
//  it biases DuckDB slower; the number is therefore a RELATIVE, PRELIMINARY
//  indicator (Mac), never an absolute "Nx faster than DuckDB" headline. The
//  credible absolute comparison comes from the x86 box with an execution-only
//  harness at M5. Correctness of this exact query is covered by the WP-8/WP-9
//  plan differential; here we only ratio wall-clock.
//
//  Determinism: seed printed; replays from one command.
//  Usage: bench_engine_vs_duckdb [--n N] [--iters I] [--seed S] [--host TAG]
//                                [--isa TAG] [--outdir DIR]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bench_harness.h"
#include "bench_report.h"
#include "bench_util.h"
#include "validity_gate.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/operator.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/result_set.h"
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

std::uint64_t drainChecksum(Operator& op) {
    op.open();
    std::uint64_t cs = 0;
    while (auto b = op.next()) cs += b->row_count;
    op.close();
    return cs;
}

// A representative analytical query over one table: filter -> GROUP BY key with
// COUNT/SUM/MIN/MAX -> ORDER BY key. Values are magnitude-bounded so every
// per-group SUM fits I64 (DuckDB's HUGEINT SUM is cast to BIGINT in the rendered
// SQL — the WP-5 overflow contract), keeping the two backends comparable.
Table makeTable(std::size_t n, std::uint64_t seed) {
    Schema s;
    s.fields.emplace_back("k", Type::I64);  // low-cardinality group key
    s.fields.emplace_back("v", Type::I64);  // bounded value
    XorShift64 rng(seed ^ 0xD0CDBu);
    std::vector<OwnedColumn> cols;
    OwnedColumn kc = OwnedColumn::make(Type::I64, n);
    OwnedColumn vc = OwnedColumn::make(Type::I64, n);
    auto* kd = reinterpret_cast<std::int64_t*>(kc.mutable_data());
    auto* vd = reinterpret_cast<std::int64_t*>(vc.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        kd[i] = static_cast<std::int64_t>(rng.next() % 64);
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
        .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(1, "tot"),
                         AggSpec::min(1, "lo"), AggSpec::max(1, "hi")})
        .sort(std::vector<SortBy>{SortBy{0}})
        .plan();
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = static_cast<std::size_t>(std::strtoull(
        argValue(argc, argv, "--n", "200000").c_str(), nullptr, 10));
    const int iters = static_cast<int>(std::strtol(
        argValue(argc, argv, "--iters", "40").c_str(), nullptr, 10));
    const std::uint64_t seed = std::strtoull(
        argValue(argc, argv, "--seed", "20260615").c_str(), nullptr, 10);
    const std::string outdir = argValue(argc, argv, "--outdir", "");
    const std::string label = "engine_vs_duckdb";

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

    std::fprintf(stderr,
                 "seed=%llu (replay: bench_engine_vs_duckdb --n %zu --iters %d "
                 "--seed %llu)\n",
                 static_cast<unsigned long long>(seed), n, iters,
                 static_cast<unsigned long long>(seed));

    const GateVerdict gate = evaluateGate(machine);
    if (!gate.accepted) {
        std::fprintf(stderr, "VALIDITY GATE: REJECTED:\n");
        for (const auto& r : gate.reasons)
            std::fprintf(stderr, "    - %s\n", r.c_str());
    }

    Table t = makeTable(n, seed);
    const qe::plan::Plan plan = makePlan(t);

    qe::metrics::LatencyHistogram heng, hduck;
    std::uint64_t checksum = 0;
    const bool haveDuck = qe::oracle::duckdb_available();

    // Warmup both backends (page-in, DuckDB catalog init) — excluded.
    { auto tree = plan.lower(); checksum ^= drainChecksum(*tree); }
    if (haveDuck) {
        try {
            checksum ^= qe::oracle::run_plan_duckdb(plan).rows.size();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "DuckDB warmup raised: %s\n", e.what());
        }
    }

    for (int it = 0; it < iters; ++it) {
        std::uint64_t t0 = nowNs();
        { auto tree = plan.lower(); checksum ^= drainChecksum(*tree); }
        std::uint64_t t1 = nowNs();
        heng.record(t1 - t0);

        if (haveDuck) {
            t0 = nowNs();
            const auto rs = qe::oracle::run_plan_duckdb(plan);
            t1 = nowNs();
            hduck.record(t1 - t0);
            checksum ^= rs.rows.size();
        }
    }

    const double eng_p50 = static_cast<double>(heng.valueAtQuantile(0.50));
    const double duck_p50 =
        haveDuck ? static_cast<double>(hduck.valueAtQuantile(0.50)) : 0.0;

    JsonWriter w;
    w.beginObject();
    writeRunTags(
        w, "bench_engine_vs_duckdb", label, host, isa, seed, preliminary,
        preliminary
            ? "PRELIMINARY / RELATIVE-ONLY (macOS, S2): ratio includes DuckDB's "
              "per-call table ingest (the only oracle seam); no absolute headline."
            : "engine-vs-duckdb relative ratio (note: includes DuckDB ingest)");
    w.key("config");
    w.beginObject();
    w.kv("rows", static_cast<std::uint64_t>(n));
    w.kv("iters", iters);
    w.kv("checksum", checksum);
    w.kv("query", plan.to_string());
    w.endObject();
    w.key("machine_start");
    writeMachine(w, machine);
    w.key("validity");
    writeGateVerdict(w, gate);
    w.key("results");
    w.beginObject();
    w.kv("valid", gate.accepted);
    w.kv("duckdb_available", haveDuck);
    w.key("engine_ns");
    writeHistogram(w, heng);
    if (haveDuck) {
        w.key("duckdb_ns");
        writeHistogram(w, hduck);
        w.kv("speedup_p50_duckdb_over_engine",
             eng_p50 > 0 ? duck_p50 / eng_p50 : 0.0);
        w.kv("speedup_p99_duckdb_over_engine",
             speedupAt(hduck, heng, 0.99));
    } else {
        w.kv("note_no_duckdb",
             "DuckDB not staged (QE_WITH_DUCKDB off): engine timings only. Stage "
             "third_party/duckdb/ per VENDORING.md for the ratio.");
    }
    w.endObject();
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    if (haveDuck)
        std::fprintf(stderr,
                     "engine p50=%.0fus | duckdb p50=%.0fus | duckdb/engine "
                     "p50=%.2fx | valid=%s\n",
                     eng_p50 / 1e3, duck_p50 / 1e3,
                     eng_p50 > 0 ? duck_p50 / eng_p50 : 0.0,
                     gate.accepted ? "true" : "false");
    else
        std::fprintf(stderr,
                     "engine p50=%.0fus | DuckDB NOT staged (no ratio)\n",
                     eng_p50 / 1e3);

    if (!outdir.empty()) {
        const std::string path =
            outdir + "/" + label + "_" + host + "_" + isa + ".json";
        if (!writeFile(path, w.str() + "\n"))
            std::fprintf(stderr, "failed to write %s\n", path.c_str());
    }
    return gate.accepted ? 0 : 3;
}
