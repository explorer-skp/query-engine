//  WP-9: THE MUTATION CATALOG — one documented registry that consolidates every
//  must-flag mutant the engine ships. "A checker that cannot fail proves nothing"
//  (RIGOR.md rule 4): the oracle/differential layer is only credible because we
//  can enumerate deliberately-broken engine variants and SHOW the suite flags each
//  one while the un-mutated engine passes the same diffs.
//
//  Until WP-9 each work package shipped its own mutants in isolation (simd/expr/
//  ops/plan *_mutants.* + a per-WP *_mutation_test.cpp). This registry names each
//  one in ONE place — the §12 hazard it represents, the planted-mutant symbol, and
//  the test that catches it — and gives every entry a runnable check() that
//  reproduces the decisive clean-vs-mutant comparison and returns a Verdict. The
//  meta-test (tests/mutation_catalog_meta_test.cpp) iterates the registry and
//  asserts (a) >= kMinCatalogEntries entries, (b) every named §12 hazard class is
//  spanned, and (c) for EVERY entry the un-mutated path passes and the mutant is
//  flagged. The full inventory + prose lives in MUTATION_CATALOG.md.
//
//  Each check() is DETERMINISTIC (fixed scenarios / a fixed catalog seed) so the
//  meta-test never flakes; the seeded FUZZ generators that replay from --seed N are
//  a separate surface (oracle/generators.*, exercised by the differential tests).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace qe::catalog {

// The §12 hazard classes the catalog must span (the "Planted-bug hotspots" list in
// the project design notes, plus the two WP-9 carry-forwards). Used by the meta-test
// to assert coverage, not just a raw entry count.
enum class Hazard : std::uint8_t {
    kSimdTailRemainder,        // SIMD tail/remainder past the last full vector
    kNullPropagation,          // three-valued (Kleene) null propagation
    kAllValidFastPath,         // all-valid fast path skipping a genuine null
    kHashProbeOverflow,        // hash probe walk / capacity-growth rehash
    kAggregationOverflow,      // aggregation integer (accumulator) overflow
    kSelectionVectorAliasing,  // selection-vector index aliasing after compaction
    kFloatToIntRound,          // carry-forward: float->int round via +0.5 (WP-2)
    kOperatorOrchestration,    // operator/plan wiring (join/sort/agg/lowering)
};

const char* hazard_label(Hazard h);  // human §12 label, for reports

// The result of running one catalog entry's check(): the un-mutated path must be
// ACCEPTED (clean_passes) and the mutated path must be REJECTED (mutant_flagged).
// An entry is "good" iff both hold.
struct Verdict {
    bool clean_passes = false;
    bool mutant_flagged = false;
    std::string detail;  // human note: the diff message / values that diverged

    bool good() const { return clean_passes && mutant_flagged; }
};

// One catalog entry. `check` is self-contained: it builds the clean and mutated
// scenarios, runs the SAME checker that catches the mutant in CI, and reports the
// verdict.
struct Entry {
    std::string id;             // stable slug, e.g. "simd_tail_remainder"
    Hazard hazard;              // §12 hazard class
    std::string mutant;         // the planted-mutant symbol this entry exercises
    std::string catching_test;  // the CI test that catches it
    std::function<Verdict()> check;
};

// The minimum the brief requires (>= 6). The registry ships more; this is the gate
// the meta-test asserts.
inline constexpr std::size_t kMinCatalogEntries = 6;

// The consolidated registry (stable order). Built once on first call.
const std::vector<Entry>& catalog();

}  // namespace qe::catalog
