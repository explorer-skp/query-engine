// ported from raft-rsm/bench (host-capture driver) — rigor infra, Raft-specifics stripped
//
// Emits the host-validity record as JSON: the "is this machine fit to
// benchmark on, and how was the binary built" document that every later
// measurement must travel with. Detects topology/ISA at runtime; never
// hardcodes core count, width, or arch.
//
// Usage: host_state [--host TAG] [--isa TAG]
//   --host overrides the host label (default derived from the CPU brand)
//   --isa  overrides the ISA label (default the compiled-for ISA)
//
// On macOS the timing validity is reported preliminary/relative-only, per the
// project's host-honesty rule (no governor, no pinning, P/E heterogeneity).

#include <cctype>
#include <cstdio>
#include <string>
#include <string_view>

#include "bench_report.h"

using namespace qe::bench;

namespace {

// Derives a default host tag from the CPU brand, e.g. "Apple M4 Pro" ->
// "mac-m4". Falls back to a generic tag when no Apple-silicon marker is found.
std::string deriveHostTag(const std::string& brand) {
#if defined(__APPLE__)
    const auto pos = brand.find('M');
    // Look for "M<digit>" (the Apple-silicon generation marker).
    for (std::size_t i = pos; i != std::string::npos && i + 1 < brand.size();
         i = brand.find('M', i + 1)) {
        if (std::isdigit(static_cast<unsigned char>(brand[i + 1]))) {
            std::string gen = "m";
            std::size_t j = i + 1;
            while (j < brand.size() &&
                   std::isdigit(static_cast<unsigned char>(brand[j]))) {
                gen += brand[j++];
            }
            return "mac-" + gen;
        }
    }
    return "mac";
#else
    (void)brand;
    return "linux-host";
#endif
}

std::string argValue(int argc, char** argv, std::string_view flag) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    const MachineState m = captureMachineState();

    std::string host = argValue(argc, argv, "--host");
    if (host.empty()) host = deriveHostTag(m.cpuModel);
    std::string isa = argValue(argc, argv, "--isa");
    if (isa.empty()) isa = m.isa;

#if defined(__APPLE__)
    const char* timingValidity = "preliminary-relative-only";
#else
    const char* timingValidity = "host-governed";  // x86 box: real governor/pin
#endif

    JsonWriter w;
    w.beginObject();
    w.kv("kind", "host_state");
    w.kv("host", host);
    w.kv("isa", isa);
    w.kv("timing_validity", timingValidity);
    w.key("machine");
    writeMachine(w, m);
    w.endObject();

    std::printf("%s\n", w.str().c_str());
    return 0;
}
