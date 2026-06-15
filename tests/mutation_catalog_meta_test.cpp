//  WP-9 META-TEST — the catalog is the credibility layer's own checker, so it too
//  must be unable to silently pass. This test iterates the consolidated mutation
//  catalog (tests/mutation_catalog.*) and asserts, in ONE place:
//
//    1. the catalog has >= kMinCatalogEntries entries (the brief's >= 6 floor);
//    2. every named §12 hazard class is SPANNED (not just a raw count);
//    3. for EVERY entry: the un-mutated engine PASSES the diff (clean_passes) AND
//       the planted mutant is FLAGGED (mutant_flagged).
//
//  Point 3 is the whole game: a differential that cannot fail proves nothing, so we
//  prove it fails on each mutant — while staying green on the real engine.
//
//  When QE_WITH_DUCKDB is set, the differential-backed entries escalate to the
//  AUTHORITATIVE DuckDB golden model (each entry's check() runs DuckDB itself), so
//  "flagged" / "passes" are decided against DuckDB, not just the reference oracle.
//
//  Deterministic by construction (fixed scenarios / a fixed catalog seed); the
//  seed printed by the harness is for uniformity with the rest of the suite. There
//  is nothing to replay — but `./mutation_catalog_meta_test` reproduces it exactly.

#include <set>
#include <string>

#include "doctest/doctest.h"

#include "tests/mutation_catalog.h"

using namespace qe::catalog;

TEST_CASE("catalog: at least the required number of entries") {
    REQUIRE(catalog().size() >= kMinCatalogEntries);
    MESSAGE("catalog entries: " << catalog().size()
                                << " (floor " << kMinCatalogEntries << ")");
}

TEST_CASE("catalog: every named §12 hazard class is spanned") {
    std::set<Hazard> seen;
    for (const Entry& e : catalog()) seen.insert(e.hazard);

    // The brief's hotspot list + the two WP-9 carry-forwards, all required.
    const Hazard required[] = {
        Hazard::kSimdTailRemainder,   Hazard::kNullPropagation,
        Hazard::kAllValidFastPath,    Hazard::kHashProbeOverflow,
        Hazard::kAggregationOverflow, Hazard::kSelectionVectorAliasing,
        Hazard::kFloatToIntRound,     Hazard::kOperatorOrchestration,
    };
    for (const Hazard h : required) {
        CHECK_MESSAGE(seen.count(h) == 1, "hazard not spanned: " << hazard_label(h));
    }
}

TEST_CASE("catalog: ids are unique (no duplicate registry entries)") {
    std::set<std::string> ids;
    for (const Entry& e : catalog()) {
        const bool fresh = ids.insert(e.id).second;
        CHECK_MESSAGE(fresh, "duplicate catalog id: " << e.id);
    }
}

// THE gate: every entry's un-mutated path passes and its mutant is flagged.
TEST_CASE("catalog: un-mutated engine passes AND every mutant is flagged") {
    for (const Entry& e : catalog()) {
        CAPTURE(e.id);
        const Verdict v = e.check();
        MESSAGE("[" << e.id << "] mutant=" << e.mutant << " | catching_test="
                    << e.catching_test << " | clean_passes=" << v.clean_passes
                    << " mutant_flagged=" << v.mutant_flagged
                    << " | " << std::string(hazard_label(e.hazard))
                    << (v.detail.empty() ? "" : (" | " + v.detail)));
        CHECK_MESSAGE(v.clean_passes,
                      "UN-MUTATED engine failed its own diff for " << e.id);
        CHECK_MESSAGE(v.mutant_flagged,
                      "MUTANT NOT FLAGGED (checker cannot fail!) for " << e.id);
    }
}
