// WP-0: runtime Highway-target probe (provenance helper, not an engine kernel).
//
// Returns the SIMD target Google Highway dispatches to on THIS machine, detected
// at runtime — the honest value for the provenance/host-state JSON, replacing the
// WP-H compile-time "n/a-WP-H" placeholder. Lives in tools/ (linked only by the
// JSON-emitting drivers) so the shared bench_infra library and the WP-H self-tests
// stay Highway-free.
#pragma once

#include <string>

namespace qe::tools {

// e.g. "NEON" on Apple Silicon, "AVX3"/"AVX3_ZEN4"/"AVX2" on x86. Never hardcoded:
// the value comes from Highway's runtime dispatch.
std::string dispatchedHighwayTarget();

}  // namespace qe::tools
