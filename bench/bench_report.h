// ported from raft-rsm/bench/bench_report.h — rigor infra, Raft-specifics stripped
#pragma once

// Result reporting: machine-state capture (the "is this host fit to benchmark
// on" record that travels WITH the numbers) and a minimal JSON writer for the
// raw per-run data files that the plots and README tables regenerate from.
//
// This is the validity-gate input later work packages read: every measurement
// carries the host it came from, the ISA it was compiled for, the build flags,
// the Highway target, and the git commit — so a number can never drift from
// the conditions that produced it.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "histogram.h"

namespace qe::bench {

struct MachineState {
    // --- CPU topology (detected at runtime; never hardcoded) ---
    std::string cpuModel;
    int logicalCpus = 0;
    int physicalCpus = 0;     // macOS: hw.physicalcpu; Linux: best-effort (0=n/a)
    int perfCores = 0;        // macOS: hw.perflevel0.logicalcpu (P cores)
    int effCores = 0;         // macOS: hw.perflevel1.logicalcpu (E cores)
    std::uint64_t ramBytes = 0;

    // --- Thermal / frequency governance (Linux-rich; macOS reports n/a) ---
    std::string governors;    // distinct scaling governors, or "n/a-macOS"
    std::string noTurbo;      // intel_pstate no_turbo ("0"/"1"/"n/a")
    std::string curFreqMHz;   // "min-max" across policies, or "n/a"
    long maxTempC = -1;       // hottest thermal zone, degrees C (-1: n/a)

    std::string kernel;
    double loadavg1 = 0.0;

    // --- Build provenance (compile-time; ISA detected from compiler macros) ---
    std::string isa;          // "neon" / "avx512" / "avx2" / "sse" / "scalar"
    std::string compiler;     // e.g. "AppleClang 17.0.0"
    std::string cxxFlags;     // optimization + arch + sanitizer summary
    std::string hwyTarget;    // Highway dispatch target, or "n/a-WP-H"
    std::string gitCommit;    // short hash, or "uncommitted"
};

MachineState captureMachineState();

// Minimal JSON emitter (no dependency; the engine's from-scratch budget stays
// untouched). Comma/nesting handled internally; strings escaped.
class JsonWriter {
public:
    void beginObject() { container('{'); }
    void endObject() { close('}'); }
    void beginArray() { container('['); }
    void endArray() { close(']'); }

    void key(std::string_view k) {
        comma();
        appendString(k);
        out_ += ':';
        if (!needComma_.empty()) needComma_.back() = false;
    }

    void value(std::string_view v) {
        comma();
        appendString(v);
    }
    void value(const char* v) { value(std::string_view(v)); }
    void value(bool b) {
        comma();
        out_ += b ? "true" : "false";
    }
    void value(std::uint64_t v);
    void value(int v) { value(static_cast<std::uint64_t>(v)); }
    void value(double v);

    template <typename T>
    void kv(std::string_view k, T v) {
        key(k);
        value(v);
    }

    const std::string& str() const { return out_; }

private:
    void comma() {
        if (!needComma_.empty()) {
            if (needComma_.back()) out_ += ',';
            needComma_.back() = true;
        }
    }
    void container(char open) {
        comma();
        out_ += open;
        needComma_.push_back(false);
    }
    void close(char c) {
        out_ += c;
        needComma_.pop_back();
        if (!needComma_.empty()) needComma_.back() = true;
    }
    void appendString(std::string_view s);

    std::string out_;
    std::vector<bool> needComma_;
};

// Standard histogram block: count/min/mean/percentiles/max plus a quantile
// grid dense enough to regenerate a CDF plot from the saved file.
void writeHistogram(JsonWriter& w, const qe::metrics::LatencyHistogram& h);

void writeMachine(JsonWriter& w, const MachineState& m);

// Writes the JSON document to `path` (returns false on I/O failure).
bool writeFile(const std::string& path, const std::string& contents);

}  // namespace qe::bench
