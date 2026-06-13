// ported from raft-rsm/bench/bench_report.cpp — rigor infra, Raft-specifics stripped
#include "bench_report.h"

#include <sys/utsname.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// --- Compile-time provenance (CMake injects these; fall back if absent) ---
#ifndef QE_GIT_COMMIT
#define QE_GIT_COMMIT "uncommitted"
#endif
#ifndef QE_CXX_COMPILER
#define QE_CXX_COMPILER "unknown-compiler"
#endif
#ifndef QE_CXX_FLAGS
#define QE_CXX_FLAGS "n/a"
#endif
#ifndef QE_HWY_TARGET
#define QE_HWY_TARGET "n/a-WP-H"
#endif

namespace qe::bench {

namespace {

// Used only by the Linux procfs/sysfs branch; absent from the macOS path.
[[maybe_unused]] std::string readFirstLine(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f && std::getline(f, line)) return line;
    return {};
}

// The ISA the binary was actually compiled for — detected, never hardcoded.
// This is what makes a measurement honest about which kernel path ran.
std::string detectIsa() {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__SSE2__)
    return "sse";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "neon";
#else
    return "scalar";
#endif
}

void fillProvenance(MachineState& m) {
    m.isa = detectIsa();
    m.compiler = QE_CXX_COMPILER;
    m.cxxFlags = QE_CXX_FLAGS;
    m.hwyTarget = QE_HWY_TARGET;
    m.gitCommit = QE_GIT_COMMIT;
}

#if defined(__APPLE__)
std::string sysctlStr(const char* name) {
    std::size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) {
        return {};
    }
    std::string buf(len, '\0');
    if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0) return {};
    while (!buf.empty() && buf.back() == '\0') buf.pop_back();
    return buf;
}

// Returns -1 when the key is absent (e.g. perflevel keys on a uniform-core
// Intel Mac), so callers can treat "absent" distinctly from "zero".
long long sysctlInt(const char* name) {
    long long v = 0;
    std::size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return -1;
    return v;
}
#endif

}  // namespace

MachineState captureMachineState() {
    MachineState m;

#if defined(__APPLE__)
    // ---- macOS / Apple Silicon branch (sysctl) ----
    // Frequency governors, turbo, and per-policy cur-freq are Linux cpufreq
    // concepts with no macOS equivalent: report them honestly as n/a so the
    // validity gate never mistakes "unmeasurable here" for "measured nominal".
    m.cpuModel = sysctlStr("machdep.cpu.brand_string");
    if (m.cpuModel.empty()) m.cpuModel = "unknown";

    const long long logical = sysctlInt("hw.logicalcpu");
    const long long physical = sysctlInt("hw.physicalcpu");
    m.logicalCpus = logical > 0 ? static_cast<int>(logical) : 0;
    m.physicalCpus = physical > 0 ? static_cast<int>(physical) : 0;

    // P/E split: perflevel0 = performance cores, perflevel1 = efficiency.
    const long long pCores = sysctlInt("hw.perflevel0.logicalcpu");
    const long long eCores = sysctlInt("hw.perflevel1.logicalcpu");
    m.perfCores = pCores > 0 ? static_cast<int>(pCores) : 0;
    m.effCores = eCores > 0 ? static_cast<int>(eCores) : 0;

    const long long mem = sysctlInt("hw.memsize");
    m.ramBytes = mem > 0 ? static_cast<std::uint64_t>(mem) : 0;

    m.governors = "n/a-macOS";   // no cpufreq governor on Darwin
    m.noTurbo = "n/a";           // no intel_pstate knob on Darwin
    m.curFreqMHz = "n/a";        // hw.cpufrequency is gone on Apple Silicon
    m.maxTempC = -1;             // SMC thermal needs IOKit/entitlements: n/a

    double la[3] = {0, 0, 0};
    if (getloadavg(la, 3) > 0) m.loadavg1 = la[0];
#else
    // ---- Linux branch (procfs + sysfs), preserved from the source ----
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("processor", 0) == 0) ++m.logicalCpus;
            if (m.cpuModel.empty() && line.rfind("model name", 0) == 0) {
                const auto colon = line.find(':');
                if (colon != std::string::npos) {
                    m.cpuModel = line.substr(colon + 2);
                }
            }
        }
    }
    {
        std::set<std::string> governors;
        long minKHz = -1;
        long maxKHz = -1;
        std::error_code ec;
        const std::string base = "/sys/devices/system/cpu/cpufreq";
        for (const auto& entry :
             std::filesystem::directory_iterator(base, ec)) {
            const auto gov =
                readFirstLine(entry.path().string() + "/scaling_governor");
            if (!gov.empty()) governors.insert(gov);
            const auto cur =
                readFirstLine(entry.path().string() + "/scaling_cur_freq");
            if (!cur.empty()) {
                const long khz = std::stol(cur);
                if (minKHz < 0 || khz < minKHz) minKHz = khz;
                if (khz > maxKHz) maxKHz = khz;
            }
        }
        std::string joined;
        for (const auto& g : governors) {
            if (!joined.empty()) joined += ",";
            joined += g;
        }
        m.governors = joined.empty() ? "n/a" : joined;
        if (minKHz > 0) {
            m.curFreqMHz = std::to_string(minKHz / 1000) + "-" +
                           std::to_string(maxKHz / 1000);
        } else {
            m.curFreqMHz = "n/a";
        }
    }
    {
        const auto v =
            readFirstLine("/sys/devices/system/cpu/intel_pstate/no_turbo");
        m.noTurbo = v.empty() ? "n/a" : v;
    }
    {
        const auto la = readFirstLine("/proc/loadavg");
        if (!la.empty()) m.loadavg1 = std::stod(la);
    }
    {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(
                 "/sys/class/thermal", ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("thermal_zone", 0) != 0) continue;
            const auto t = readFirstLine(entry.path().string() + "/temp");
            if (!t.empty()) {
                const long c = std::stol(t) / 1000;
                if (c > m.maxTempC) m.maxTempC = c;
            }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string label;
        std::uint64_t kb = 0;
        if (f >> label >> kb && label == "MemTotal:") {
            m.ramBytes = kb * 1024;
        }
    }
#endif

    {
        utsname uts{};
        if (uname(&uts) == 0) {
            m.kernel = std::string(uts.sysname) + " " + uts.release;
        }
    }

    fillProvenance(m);
    return m;
}

void JsonWriter::value(std::uint64_t v) {
    comma();
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%llu",
                  static_cast<unsigned long long>(v));
    out_ += buf;
}

void JsonWriter::value(double v) {
    comma();
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    out_ += buf;
}

void JsonWriter::appendString(std::string_view s) {
    out_ += '"';
    for (const char c : s) {
        switch (c) {
            case '"': out_ += "\\\""; break;
            case '\\': out_ += "\\\\"; break;
            case '\n': out_ += "\\n"; break;
            case '\t': out_ += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out_ += buf;
                } else {
                    out_ += c;
                }
        }
    }
    out_ += '"';
}

void writeHistogram(JsonWriter& w, const qe::metrics::LatencyHistogram& h) {
    w.beginObject();
    w.kv("count", h.count());
    w.kv("min_ns", h.minValue());
    w.kv("mean_ns", h.mean());
    w.kv("p50_ns", h.valueAtQuantile(0.50));
    w.kv("p90_ns", h.valueAtQuantile(0.90));
    w.kv("p99_ns", h.valueAtQuantile(0.99));
    w.kv("p999_ns", h.valueAtQuantile(0.999));
    w.kv("p9999_ns", h.valueAtQuantile(0.9999));
    w.kv("max_ns", h.maxValue());
    // Dense quantile grid: enough to regenerate the CDF plot from the file.
    w.key("quantiles");
    w.beginArray();
    const auto point = [&](double q) {
        w.beginArray();
        w.value(q);
        w.value(h.valueAtQuantile(q));
        w.endArray();
    };
    for (int i = 1; i <= 99; ++i) point(i / 100.0);
    for (const double q :
         {0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 1.0}) {
        point(q);
    }
    w.endArray();
    w.endObject();
}

void writeMachine(JsonWriter& w, const MachineState& m) {
    w.beginObject();
    w.kv("cpu_model", m.cpuModel);
    w.kv("logical_cpus", m.logicalCpus);
    w.kv("physical_cpus", m.physicalCpus);
    w.kv("perf_cores", m.perfCores);
    w.kv("eff_cores", m.effCores);
    w.kv("ram_bytes", m.ramBytes);
    w.kv("governors", m.governors);
    w.kv("no_turbo", m.noTurbo);
    w.kv("cur_freq_mhz", m.curFreqMHz);
    w.kv("max_temp_c", static_cast<std::uint64_t>(
                           m.maxTempC < 0 ? 0 : m.maxTempC));
    w.kv("kernel", m.kernel);
    w.kv("loadavg1", m.loadavg1);
    // Build provenance — the half of the validity record that says HOW the
    // number was produced, not just on what host.
    w.kv("isa", m.isa);
    w.kv("compiler", m.compiler);
    w.kv("cxx_flags", m.cxxFlags);
    w.kv("highway_target", m.hwyTarget);
    w.kv("git_commit", m.gitCommit);
    w.endObject();
}

bool writeFile(const std::string& path, const std::string& contents) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    f << contents;
    return static_cast<bool>(f);
}

}  // namespace qe::bench
