//  WP-10: shared harness helpers (non-template). See bench_harness.h.
#include "bench_harness.h"

namespace qe::bench {

Throughput throughputFrom(std::uint64_t rows, std::uint64_t bytesScanned,
                          double latencyNs) {
    Throughput t;
    t.bytes_scanned = bytesScanned;
    if (latencyNs > 0.0) {
        const double secs = latencyNs / 1e9;
        t.rows_per_s = static_cast<double>(rows) / secs;
        t.gb_per_s = (static_cast<double>(bytesScanned) / 1e9) / secs;
    }
    return t;
}

Roofline classifyRoofline(std::uint64_t flops, std::uint64_t bytes,
                          double ridge_flops_per_byte, bool preliminary) {
    Roofline r;
    r.flops = flops;
    r.bytes = bytes;
    r.arithmetic_intensity =
        bytes > 0 ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;
    r.ridge_flops_per_byte = ridge_flops_per_byte;
    r.compute_bound = r.arithmetic_intensity > ridge_flops_per_byte;
    r.preliminary = preliminary;
    return r;
}

double speedupAt(const qe::metrics::LatencyHistogram& slow,
                 const qe::metrics::LatencyHistogram& fast, double q) {
    const double s = static_cast<double>(slow.valueAtQuantile(q));
    const double f = static_cast<double>(fast.valueAtQuantile(q));
    return f > 0.0 ? s / f : 0.0;
}

namespace {

void writeRoofline(JsonWriter& w, const Roofline& r) {
    w.beginObject();
    w.kv("flops", r.flops);
    w.kv("bytes", r.bytes);
    w.kv("arithmetic_intensity_flops_per_byte", r.arithmetic_intensity);
    w.kv("ridge_point_flops_per_byte", r.ridge_flops_per_byte);
    w.kv("classification", r.compute_bound ? "compute-bound" : "bandwidth-bound");
    w.kv("preliminary", r.preliminary);
    w.kv("note", r.preliminary
                     ? "PRELIMINARY: ridge point ASSUMED, not measured on this "
                       "host (no credible peak-FLOP/peak-BW roofline on macOS, "
                       "RIGOR.md S2); the headline roofline comes from x86."
                     : "ridge point from host machine balance");
    w.endObject();
}

}  // namespace

void writeRunTags(JsonWriter& w, std::string_view kind, std::string_view label,
                  std::string_view host, std::string_view isa,
                  std::uint64_t seed, bool preliminary, std::string_view note) {
    w.kv("kind", kind);
    w.kv("label", label);
    w.kv("host", host);
    w.kv("isa", isa);
    w.kv("seed", seed);
    w.kv("preliminary", preliminary);
    w.kv("note", note);
}

void writeVecScalarDoc(JsonWriter& w, std::string_view label, std::string_view op,
                       std::string_view host, std::string_view isa,
                       std::uint64_t seed, bool preliminary,
                       std::string_view note, int iters, std::uint64_t checksum,
                       const MachineState& machine, const GateVerdict& gate,
                       const VecScalarResult& r) {
    const double vec_p50 = static_cast<double>(r.vec.valueAtQuantile(0.50));
    const double sca_p50 = static_cast<double>(r.scalar.valueAtQuantile(0.50));
    const Throughput vec_thr =
        throughputFrom(r.rows, r.bytes_scanned, vec_p50);
    const Throughput sca_thr =
        throughputFrom(r.rows, r.bytes_scanned, sca_p50);
    const bool valid = gate.accepted;

    w.beginObject();
    writeRunTags(w, "bench_vec_scalar", label, host, isa, seed, preliminary, note);
    w.kv("operator", op);

    w.key("config");
    w.beginObject();
    w.kv("rows", r.rows);
    w.kv("bytes_scanned_per_call", r.bytes_scanned);
    w.kv("iters", iters);
    w.kv("checksum", checksum);
    w.endObject();

    w.key("machine_start");
    writeMachine(w, machine);
    w.key("validity");
    writeGateVerdict(w, gate);

    w.key("results");
    w.beginObject();
    w.kv("valid", valid);
    w.key("vec_ns");
    writeHistogram(w, r.vec);
    w.key("scalar_ns");
    writeHistogram(w, r.scalar);
    w.kv("speedup_p50_scalar_over_vec", speedupAt(r.scalar, r.vec, 0.50));
    w.kv("speedup_p99_scalar_over_vec", speedupAt(r.scalar, r.vec, 0.99));
    w.kv("speedup_p999_scalar_over_vec", speedupAt(r.scalar, r.vec, 0.999));
    w.key("throughput");
    w.beginObject();
    w.kv("vec_rows_per_s", vec_thr.rows_per_s);
    w.kv("scalar_rows_per_s", sca_thr.rows_per_s);
    w.kv("vec_gb_per_s", vec_thr.gb_per_s);
    w.kv("scalar_gb_per_s", sca_thr.gb_per_s);
    w.kv("bytes_scanned_per_call", r.bytes_scanned);
    w.endObject();
    w.key("roofline");
    writeRoofline(w, r.roofline);
    w.endObject();  // results
    w.endObject();  // doc
}

}  // namespace qe::bench
