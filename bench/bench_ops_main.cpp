//  WP-10: the VEC-VS-SCALAR per-operator benchmark driver — the engine's relative
//  speed story, proven the only way host variance cancels: drive the SAME input,
//  same seed, through each operator's vector path and its independently-written
//  SCALAR TWIN, and report the RATIO of the two (RIGOR.md rule 7 / §2). No new
//  kernels are written here — every path is reached through a seam the operator
//  ALREADY exposes:
//      * expr      — evaluate(..., Backend::{Vector,Scalar})
//      * hashtable — find(..., HashPath::{kVector,kScalar})
//      * aggregate — Aggregate::set_paths(.., AggKernelPath::{kVector,kScalar})
//      * sort      — Sort::set_gather_path(GatherPath::{kVector,kScalar})
//
//  Before any number is reported, the VALIDITY GATE (bench/validity_gate.h) reads
//  this host's state and a quiescence probe and REJECTS a contaminated run: the
//  doc is still written, but with results.valid=false, and stderr says REJECTED.
//  `--inject-load` deliberately loads the machine to demonstrate the gate biting.
//
//  HOST HONESTY (§2): every doc is tagged host+isa; Mac docs carry
//  "preliminary":true; the roofline ridge point is ASSUMED on Mac (labeled). The
//  SAME driver reruns on the x86 box unchanged — nothing here hardcodes width,
//  cache, core count, or ISA (topology comes from runtime host state; the kernels
//  pick their own width).
//
//  Determinism: the seed is printed and the run replays from a single command.
//  Usage: bench_ops [--op all|expr|hashtable|aggregate|sort] [--n N] [--iters I]
//                   [--seed S] [--host TAG] [--isa TAG] [--outdir DIR]
//                   [--ridge FLOPS_PER_BYTE] [--inject-load]

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "bench_harness.h"
#include "bench_report.h"
#include "bench_util.h"
#include "validity_gate.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/hashtable.h"
#include "ops/operator.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "tools/hwy_target.h"

using namespace qe;
using namespace qe::bench;
using qe::metrics::LatencyHistogram;

namespace {

std::string argValue(int argc, char** argv, std::string_view flag,
                     std::string def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}
bool argFlag(int argc, char** argv, std::string_view flag) {
    for (int i = 1; i < argc; ++i)
        if (flag == argv[i]) return true;
    return false;
}

std::string deriveHostTag(const std::string& brand) {
#if defined(__APPLE__)
    const auto pos = brand.find('M');
    for (std::size_t i = pos; i != std::string::npos && i + 1 < brand.size();
         i = brand.find('M', i + 1)) {
        if (brand[i + 1] >= '0' && brand[i + 1] <= '9') {
            std::string gen = "m";
            std::size_t j = i + 1;
            while (j < brand.size() && brand[j] >= '0' && brand[j] <= '9')
                gen += brand[j++];
            return "mac-" + gen;
        }
    }
    return "mac";
#else
    (void)brand;
    return "linux-host";
#endif
}

// Read 8 bytes at logical element `idx` of a freshly-produced column's data as a
// u64 checksum word (defeats dead-store elimination of the timed work).
std::uint64_t sampleWord(const std::byte* data, std::size_t width,
                         std::size_t idx) {
    std::uint64_t v = 0;
    std::memcpy(&v, data + idx * width, width < 8 ? width : 8);
    return v;
}

// Pull an operator dry, folding row counts and one sampled value per batch into a
// checksum (so the operator's work is materialized and cannot be elided).
std::uint64_t drainChecksum(Operator& op) {
    op.open();
    std::uint64_t cs = 0;
    while (auto b = op.next()) {
        const Batch& batch = *b;
        cs += batch.row_count;
        if (batch.row_count > 0 && !batch.cols.empty()) {
            const Column& c0 = batch.cols[0];
            const std::size_t phys =
                batch.sel ? batch.sel->idx[0] : 0;
            cs ^= sampleWord(c0.data, byte_width(c0.type), phys);
        }
    }
    op.close();
    return cs;
}

// ---- per-operator workloads -------------------------------------------------

VecScalarResult benchExpr(std::size_t n, int iters, std::uint64_t seed,
                          double ridge, bool preliminary,
                          std::uint64_t& checksum) {
    // Three dense F64 columns; expression (a*b)+c — 2 flops/row, reads 3 cols,
    // writes 1 (32 bytes/row moved).
    OwnedBatch ob;
    XorShift64 rng(seed ^ 0xE57Au);
    for (int col = 0; col < 3; ++col) {
        OwnedColumn c = OwnedColumn::make(Type::F64, n);
        auto* d = reinterpret_cast<double*>(c.mutable_data());
        for (std::size_t i = 0; i < n; ++i)
            d[i] = static_cast<double>(static_cast<std::int64_t>(rng.next() %
                                                                 2000000) -
                                       1000000) *
                   0.5;
        ob.add_column(std::move(c));
    }
    const Batch batch = ob.view();
    using namespace qe::expr;
    const Expr e = add(mul(col(Type::F64, 0), col(Type::F64, 1)),
                       col(Type::F64, 2));
    const std::size_t sidx = n ? (n / 2) : 0;

    auto run = [&](Backend backend) {
        OwnedColumn out = evaluate(e, batch, backend);
        return n ? sampleWord(out.data(), 8, sidx) : 0ull;
    };
    VecScalarResult r;
    r.rows = n;
    r.bytes_scanned = static_cast<std::uint64_t>(n) * 32;
    r.roofline = classifyRoofline(static_cast<std::uint64_t>(n) * 2,
                                  r.bytes_scanned, ridge, preliminary);
    r.vec = timeSamples(iters, 3, checksum, [&] { return run(Backend::Vector); });
    r.scalar =
        timeSamples(iters, 3, checksum, [&] { return run(Backend::Scalar); });
    return r;
}

VecScalarResult benchHashTable(std::size_t n, int iters, std::uint64_t seed,
                               double ridge, bool preliminary,
                               std::uint64_t& checksum) {
    // One I64 key column, moderate cardinality, pre-built into the table; the
    // timed seam is the PROBE (find — const, repeatable). Reads 8B key, writes 4B
    // group id per row (12 bytes/row); ~5 mixing flops/row (approx, noted).
    const std::int64_t card = static_cast<std::int64_t>(n / 8 + 1);
    OwnedColumn keys = OwnedColumn::make(Type::I64, n);
    auto* k = reinterpret_cast<std::int64_t*>(keys.mutable_data());
    XorShift64 rng(seed ^ 0x4A5Bu);
    for (std::size_t i = 0; i < n; ++i)
        k[i] = static_cast<std::int64_t>(rng.next() % static_cast<std::uint64_t>(
                                                          card > 0 ? card : 1));
    const Column kv = keys.view();
    KeyColumns kc{&kv, 1, nullptr};

    HashTable ht({Type::I64});
    std::vector<std::uint32_t> gids(n ? n : 1);
    if (n) ht.insert_or_find(kc, n, gids.data(), HashPath::kVector);

    auto run = [&](HashPath path) {
        if (n) ht.find(kc, n, gids.data(), path);
        return n ? static_cast<std::uint64_t>(gids[n / 2]) : 0ull;
    };
    VecScalarResult r;
    r.rows = n;
    r.bytes_scanned = static_cast<std::uint64_t>(n) * 12;
    r.roofline = classifyRoofline(static_cast<std::uint64_t>(n) * 5,
                                  r.bytes_scanned, ridge, preliminary);
    r.vec =
        timeSamples(iters, 3, checksum, [&] { return run(HashPath::kVector); });
    r.scalar =
        timeSamples(iters, 3, checksum, [&] { return run(HashPath::kScalar); });
    return r;
}

VecScalarResult benchAggregate(const Table& t, int iters, std::uint64_t seed,
                               double ridge, bool preliminary,
                               std::uint64_t& checksum) {
    (void)seed;
    const std::size_t n = t.num_rows();
    std::vector<AggSpec> aggs = {AggSpec::count_star("cs"), AggSpec::sum(0, "s"),
                                 AggSpec::min(0, "mn"), AggSpec::max(0, "mx"),
                                 AggSpec::avg(0, "av")};
    auto run = [&](AggKernelPath path) {
        auto scan = std::make_unique<Scan>(t, 2048);
        auto agg = std::make_unique<Aggregate>(
            std::move(scan), std::vector<std::uint32_t>{}, aggs);
        agg->set_paths(HashPath::kVector, path);
        return drainChecksum(*agg);
    };
    VecScalarResult r;
    r.rows = n;
    r.bytes_scanned = static_cast<std::uint64_t>(n) * 8;  // one F64 col read
    r.roofline = classifyRoofline(static_cast<std::uint64_t>(n) * 4,
                                  r.bytes_scanned, ridge, preliminary);
    r.vec = timeSamples(iters, 3, checksum,
                        [&] { return run(AggKernelPath::kVector); });
    r.scalar = timeSamples(iters, 3, checksum,
                           [&] { return run(AggKernelPath::kScalar); });
    return r;
}

VecScalarResult benchSort(const Table& t, int iters, std::uint64_t seed,
                          double ridge, bool preliminary,
                          std::uint64_t& checksum) {
    (void)seed;
    const std::size_t n = t.num_rows();
    std::vector<SortKey> keys = {{0, SortDir::Asc, NullOrder::Last},
                                 {1, SortDir::Asc, NullOrder::Last},
                                 {2, SortDir::Asc, NullOrder::Last}};
    auto run = [&](Sort::GatherPath gp) {
        auto scan = std::make_unique<Scan>(t, 2048);
        auto sort = std::make_unique<Sort>(std::move(scan), keys);
        sort->set_path(Sort::Path::kRadix);  // radix => the vectorized gather runs
        sort->set_gather_path(gp);
        return drainChecksum(*sort);
    };
    VecScalarResult r;
    r.rows = n;
    // Whole-row gather moves all 3 columns (4+8+8=20 bytes) read+written.
    r.bytes_scanned = static_cast<std::uint64_t>(n) * 40;
    r.roofline = classifyRoofline(static_cast<std::uint64_t>(n) * 1,
                                  r.bytes_scanned, ridge, preliminary);
    r.vec = timeSamples(iters, 2, checksum,
                        [&] { return run(Sort::GatherPath::kVector); });
    r.scalar = timeSamples(iters, 2, checksum,
                           [&] { return run(Sort::GatherPath::kScalar); });
    return r;
}

// Build the all-integer table the aggregate (col 0) and sort (cols 0..2) reuse.
Table makeIntTable(std::size_t n, std::uint64_t seed) {
    Schema s;
    s.fields.emplace_back("c0", Type::I64);
    s.fields.emplace_back("c1", Type::I64);
    s.fields.emplace_back("c2", Type::TS);
    XorShift64 rng(seed ^ 0x7AB1Eu);
    std::vector<OwnedColumn> cols;
    for (int col = 0; col < 3; ++col) {
        OwnedColumn c =
            OwnedColumn::make(col == 2 ? Type::TS : Type::I64, n);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        for (std::size_t i = 0; i < n; ++i)
            d[i] = static_cast<std::int64_t>(rng.next() % 4000000) - 2000000;
        cols.push_back(std::move(c));
    }
    return Table(s, std::move(cols));
}

struct BusyLoad {
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    void start(int n) {
        for (int i = 0; i < n; ++i)
            threads.emplace_back([this] {
                volatile std::uint64_t x = 1;
                while (!stop.load(std::memory_order_relaxed)) x = x * 6364136223846793005ull + 1;
            });
    }
    ~BusyLoad() {
        stop.store(true, std::memory_order_relaxed);
        for (auto& t : threads) t.join();
    }
};

}  // namespace

int main(int argc, char** argv) {
    const std::string op = argValue(argc, argv, "--op", "all");
    const std::size_t n = static_cast<std::size_t>(std::strtoull(
        argValue(argc, argv, "--n", "262144").c_str(), nullptr, 10));
    const int iters = static_cast<int>(std::strtol(
        argValue(argc, argv, "--iters", "200").c_str(), nullptr, 10));
    const std::uint64_t seed = std::strtoull(
        argValue(argc, argv, "--seed", "20260615").c_str(), nullptr, 10);
    const std::string outdir = argValue(argc, argv, "--outdir", "");
    const double ridge =
        std::strtod(argValue(argc, argv, "--ridge", "4.0").c_str(), nullptr);
    const bool injectLoad = argFlag(argc, argv, "--inject-load");

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
                 "seed=%llu (replay: bench_ops --op %s --n %zu --iters %d "
                 "--seed %llu)\n",
                 static_cast<unsigned long long>(seed), op.c_str(), n, iters,
                 static_cast<unsigned long long>(seed));

    // --- VALIDITY GATE: read host state, reject a contaminated run ----------
    BusyLoad load;
    if (injectLoad) {
        const int hog = machine.logicalCpus > 0 ? machine.logicalCpus : 4;
        std::fprintf(stderr,
                     "--inject-load: spawning %d busy threads to contaminate the "
                     "host (gate MUST reject)\n",
                     hog);
        load.start(hog);
        machine = captureMachineState();  // re-read load under contamination
        machine.hwyTarget = qe::tools::dispatchedHighwayTarget();
    }
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

    const bool doAll = (op == "all");
    Table intTable = makeIntTable(n, seed);  // reused by aggregate + sort

    struct Out {
        std::string label, opname;
        VecScalarResult res;
        std::uint64_t checksum;
    };
    std::vector<Out> outs;
    std::uint64_t cs = 0;

    if (doAll || op == "expr") {
        cs = 0;
        outs.push_back({"expr_fma", "expr",
                        benchExpr(n, iters, seed, ridge, preliminary, cs), cs});
    }
    if (doAll || op == "hashtable") {
        cs = 0;
        outs.push_back({"hashtable_probe", "hashtable",
                        benchHashTable(n, iters, seed, ridge, preliminary, cs),
                        cs});
    }
    if (doAll || op == "aggregate") {
        cs = 0;
        outs.push_back({"aggregate_global", "aggregate",
                        benchAggregate(intTable, iters, seed, ridge,
                                       preliminary, cs),
                        cs});
    }
    if (doAll || op == "sort") {
        cs = 0;
        outs.push_back({"sort_e2e", "sort",
                        benchSort(intTable, iters, seed, ridge, preliminary, cs),
                        cs});
    }
    if (outs.empty()) {
        std::fprintf(stderr, "unknown --op '%s'\n", op.c_str());
        return 2;
    }

    const char* note =
        preliminary
            ? "PRELIMINARY / RELATIVE-ONLY (macOS, RIGOR.md S2): vec-vs-scalar "
              "ratio + percentiles only; no absolute/AVX-512/roofline headline."
            : "vec-vs-scalar relative ratio + percentiles";

    for (const auto& o : outs) {
        JsonWriter w;
        writeVecScalarDoc(w, o.label, o.opname, host, isa, seed, preliminary,
                          note, iters, o.checksum, machine, gate, o.res);
        std::printf("%s\n", w.str().c_str());
        const double sp50 = speedupAt(o.res.scalar, o.res.vec, 0.50);
        const double sp99 = speedupAt(o.res.scalar, o.res.vec, 0.99);
        std::fprintf(stderr,
                     "%-16s vec p50=%lluns scalar p50=%lluns | speedup(scalar/"
                     "vec) p50=%.2fx p99=%.2fx | valid=%s\n",
                     o.label.c_str(),
                     static_cast<unsigned long long>(
                         o.res.vec.valueAtQuantile(0.50)),
                     static_cast<unsigned long long>(
                         o.res.scalar.valueAtQuantile(0.50)),
                     sp50, sp99, gate.accepted ? "true" : "false");
        if (!outdir.empty()) {
            const std::string path =
                outdir + "/" + o.label + "_" + host + "_" + isa + ".json";
            if (!writeFile(path, w.str() + "\n"))
                std::fprintf(stderr, "failed to write %s\n", path.c_str());
        }
    }
    // A rejected run exits non-zero so a regen script can refuse to publish it.
    return gate.accepted ? 0 : 3;
}
