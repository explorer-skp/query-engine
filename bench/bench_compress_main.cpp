//  WP-14: compression-ratio + decode-throughput reporting (the §5 "reported"
//  numbers). Builds a seeded tick-like Table (irregular TS + Gorilla-friendly F64
//  price + I64 payload), encodes it, and emits ONE JSON document with the per-type
//  and overall COMPRESSED/UNCOMPRESSED bytes ratio and the streaming DECODE rows/s,
//  tagged host= + isa= and labeled PRELIMINARY / RELATIVE-ONLY (RIGOR.md §2 — this
//  is a Mac/NEON dev number; the credible roofline is the x86 box). Ratio +
//  throughput are point measurements reported WITH the exact command that
//  reproduces them (no bare averages; decode time is the median of repeated drains).
//
//  Determinism: the seed is printed and the run replays from a single command.
//  Usage:
//    bench_compress [--seed N] [--rows R] [--reps K] [--host TAG] [--isa TAG]
//                   [--out PATH]
//  Reproduce (Mac, preliminary):
//    cmake --build build --target bench_compress && \
//    ./build/bench_compress --seed 20260614 --rows 1000000 --reps 9

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "bench_report.h"
#include "bench_util.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/table.h"
#include "tools/hwy_target.h"
#include "tsx/compress.h"

using namespace qe;
using namespace qe::bench;

namespace {

std::string argValue(int argc, char** argv, std::string_view flag,
                     std::string def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}

// Build a tick-like table: TS (irregular increasing), F64 price (random walk —
// adjacent values share high bits, the Gorilla best case), I64 volume payload.
Table build_ticks(std::uint64_t seed, std::size_t n) {
    XorShift64 rng(seed ? seed : 1);
    Schema s;
    s.fields.emplace_back("ts", Type::TS);
    s.fields.emplace_back("px", Type::F64);
    s.fields.emplace_back("vol", Type::I64);
    OwnedColumn ts = OwnedColumn::make(Type::TS, n);
    OwnedColumn px = OwnedColumn::make(Type::F64, n);
    OwnedColumn vol = OwnedColumn::make(Type::I64, n);
    auto* pt = reinterpret_cast<std::int64_t*>(ts.mutable_data());
    auto* pp = reinterpret_cast<double*>(px.mutable_data());
    auto* pv = reinterpret_cast<std::int64_t*>(vol.mutable_data());
    std::int64_t t = 1'600'000'000'000'000'000LL;  // ns epoch-ish
    double price = 100.0;
    for (std::size_t i = 0; i < n; ++i) {
        t += 1'000'000 + static_cast<std::int64_t>(rng.next() % 500'000);  // ~ms gaps
        pt[i] = t;
        // Realistic tick moves: mostly ±1 cent (small XORs => Gorilla's best case).
        price += (static_cast<double>(rng.next() % 5) - 2.0) * 0.01;
        pp[i] = price;
        pv[i] = static_cast<std::int64_t>(rng.next() % 100'000);
    }
    std::vector<OwnedColumn> cols;
    cols.push_back(std::move(ts));
    cols.push_back(std::move(px));
    cols.push_back(std::move(vol));
    return Table(s, std::move(cols));
}

std::uint64_t drain_checksum(const tsx::CompressedTable& ct) {
    tsx::CompressedScan sc(ct);
    sc.open();
    std::uint64_t acc = 0;
    while (auto b = sc.next()) {
        const Batch& batch = *b;
        // Touch one column's bytes so the decode is not dead-code-eliminated.
        const Column& c = batch.cols[0];
        const auto* d = reinterpret_cast<const std::int64_t*>(c.data);
        for (std::size_t r = 0; r < batch.row_count; ++r) acc ^= static_cast<std::uint64_t>(d[r]);
    }
    sc.close();
    return acc;
}

const char* type_name(Type t) {
    switch (t) {
        case Type::I32: return "i32";
        case Type::I64: return "i64";
        case Type::F64: return "f64";
        case Type::BOOL: return "bool";
        case Type::TS: return "ts";
    }
    return "?";
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t seed = std::strtoull(
        argValue(argc, argv, "--seed", "20260614").c_str(), nullptr, 10);
    const std::size_t rows = static_cast<std::size_t>(std::strtoull(
        argValue(argc, argv, "--rows", "1000000").c_str(), nullptr, 10));
    const int reps = static_cast<int>(
        std::strtol(argValue(argc, argv, "--reps", "9").c_str(), nullptr, 10));
    const std::string outPath = argValue(argc, argv, "--out", "");

    std::fprintf(stderr,
                 "seed=%llu (replay: bench_compress --seed %llu --rows %zu "
                 "--reps %d)\n",
                 static_cast<unsigned long long>(seed),
                 static_cast<unsigned long long>(seed), rows, reps);

    const Table src = build_ticks(seed, rows);
    const tsx::CompressedTable ct = tsx::CompressedTable::encode(src);

    MachineState machine = captureMachineState();
    machine.hwyTarget = qe::tools::dispatchedHighwayTarget();
    std::string host = argValue(argc, argv, "--host", "");
    if (host.empty()) host =
#if defined(__APPLE__)
        "mac";
#else
        "linux-host";
#endif
    std::string isa = argValue(argc, argv, "--isa", machine.isa);

    // Decode throughput: median per-drain latency over `reps` full drains (warm).
    std::vector<std::uint64_t> lat;
    lat.reserve(static_cast<std::size_t>(reps));
    std::uint64_t checksum = 0;
    checksum ^= drain_checksum(ct);  // one warmup (untimed)
    for (int r = 0; r < reps; ++r) {
        const std::uint64_t t0 = nowNs();
        checksum ^= drain_checksum(ct);
        const std::uint64_t t1 = nowNs();
        lat.push_back(t1 - t0);
    }
    std::sort(lat.begin(), lat.end());
    const std::uint64_t med_ns = lat.empty() ? 0 : lat[lat.size() / 2];
    const double rows_per_s =
        med_ns ? static_cast<double>(rows) * 1e9 / static_cast<double>(med_ns)
               : 0.0;

    const std::size_t comp = ct.compressed_bytes();
    const std::size_t uncomp = ct.uncompressed_bytes();
    const double overall_ratio =
        comp ? static_cast<double>(uncomp) / static_cast<double>(comp) : 0.0;

    JsonWriter w;
    w.beginObject();
    w.kv("kind", "bench_compress");
    w.kv("host", host);
    w.kv("isa", isa);
    w.kv("hwy_target", machine.hwyTarget);
    w.kv("seed", seed);
    w.kv("preliminary", true);  // §2: Mac/NEON dev number, relative-only
    w.kv("note", "Mac numbers are preliminary / relative-only (RIGOR.md §2)");
    w.kv("rows", static_cast<std::uint64_t>(rows));
    w.kv("uncompressed_bytes", static_cast<std::uint64_t>(uncomp));
    w.kv("compressed_bytes", static_cast<std::uint64_t>(comp));
    w.kv("compression_ratio", overall_ratio);  // uncompressed / compressed
    w.kv("decode_rows_per_s_median", rows_per_s);
    w.kv("decode_median_ns", med_ns);
    w.kv("decode_reps", reps);
    w.key("per_column");
    w.beginArray();
    for (std::size_t c = 0; c < ct.num_columns(); ++c) {
        w.beginObject();
        w.kv("name", src.schema().fields[c].first);
        w.kv("type", type_name(src.schema().fields[c].second));
        const std::size_t cc = ct.compressed_bytes(c);
        const std::size_t uu = ct.uncompressed_bytes(c);
        w.kv("uncompressed_bytes", static_cast<std::uint64_t>(uu));
        w.kv("compressed_bytes", static_cast<std::uint64_t>(cc));
        w.kv("ratio", cc ? static_cast<double>(uu) / static_cast<double>(cc) : 0.0);
        w.endObject();
    }
    w.endArray();
    w.kv("checksum", checksum);  // guards against dead-code elimination
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    if (!outPath.empty() && !writeFile(outPath, w.str() + "\n")) {
        std::fprintf(stderr, "failed to write %s\n", outPath.c_str());
        return 1;
    }
    return 0;
}
