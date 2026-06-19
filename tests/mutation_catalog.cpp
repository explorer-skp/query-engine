//  WP-9: the consolidated mutation-catalog registry. See tests/mutation_catalog.h.
//
//  This TU only ASSEMBLES the registry — it names each must-flag mutant, its §12
//  hazard, and the test that catches it, and wires the runnable check() defined in
//  the conflict-free check TUs (tests/catalog_checks_*.cpp, declared in
//  tests/catalog_checks.h). It deliberately includes NO mutant header, so the three
//  distinct qe::mutant::Mutation enums never meet here.

#include "tests/mutation_catalog.h"

#include "tests/catalog_checks.h"

namespace qe::catalog {

const char* hazard_label(Hazard h) {
    switch (h) {
        case Hazard::kSimdTailRemainder:
            return "§12 SIMD tail/remainder past the last full vector";
        case Hazard::kNullPropagation:
            return "§12 null propagation (three-valued/Kleene logic)";
        case Hazard::kAllValidFastPath:
            return "§12 all-valid fast path skipping a genuine null";
        case Hazard::kHashProbeOverflow:
            return "§12 hash-join probe walk / capacity-growth overflow";
        case Hazard::kAggregationOverflow:
            return "§12 aggregation integer (accumulator) overflow";
        case Hazard::kSelectionVectorAliasing:
            return "§12 selection-vector index aliasing after compaction";
        case Hazard::kFloatToIntRound:
            return "§12 float->int cast rounding (+0.5) — WP-2 carry-forward";
        case Hazard::kOperatorOrchestration:
            return "§12 operator/plan orchestration (join/sort/agg/lowering)";
    }
    return "unknown";
}

namespace {

namespace ck = qe::catalog::checks;

std::vector<Entry> build_catalog() {
    return {
        {"simd_tail_remainder", Hazard::kSimdTailRemainder,
         "simd::mutant::all_valid_vec_tailbug", "validity_mutation_test",
         ck::simd_tail},
        {"null_propagation_kleene", Hazard::kNullPropagation,
         "expr::mutant::logic_and_twovalued", "expr_mutation_test",
         ck::null_propagation},
        {"all_valid_fastpath_skips_null", Hazard::kAllValidFastPath,
         "expr::mutant::propagate_nulls_and_allvalidbug", "expr_mutation_test",
         ck::all_valid_fastpath},
        {"hashjoin_probe_overflow", Hazard::kHashProbeOverflow,
         "mutant::HashTable{kFindStopEarly}", "hashtable_mutation_test",
         ck::hash_probe_overflow},
        {"hash_growth_rehash_drop", Hazard::kHashProbeOverflow,
         "mutant::HashTable{kSkipRehashOne}", "hashtable_mutation_test",
         ck::hash_growth_rehash},
        {"aggregation_overflow_i32", Hazard::kAggregationOverflow,
         "catalog::mutant::sum_i64_wrapping_i32", "mutation_catalog_meta_test",
         ck::aggregation_overflow},
        {"selection_vector_aliasing", Hazard::kSelectionVectorAliasing,
         "catalog::mutant::gather_in_place_aliased", "mutation_catalog_meta_test",
         ck::selection_aliasing},
        {"float_to_int_round_plus_half", Hazard::kFloatToIntRound,
         "catalog::mutant::cast_f64_to_i64_plus_half", "mutation_catalog_meta_test",
         ck::float_to_int_round},
        {"join_drop_probe_match", Hazard::kOperatorOrchestration,
         "mutant::HashJoin{kDropProbeMatch}", "join_mutation_test",
         ck::join_drop_match},
        {"sort_desc_as_asc", Hazard::kOperatorOrchestration,
         "mutant::Sort{kDescSortsAsc}", "sort_mutation_test", ck::sort_desc_as_asc},
        {"agg_fold_null_in_sum", Hazard::kOperatorOrchestration,
         "mutant::Aggregate{kFoldNullInSum}", "agg_mutation_test",
         ck::agg_fold_null_in_sum},
        {"plan_drop_sort", Hazard::kOperatorOrchestration,
         "plan::lower_mutant{kDropSort}", "plan_mutation_test", ck::plan_drop_sort},
        {"asof_boundary_strict", Hazard::kOperatorOrchestration,
         "tsx::mutant::AsofJoin{kBoundaryStrict}", "asof_mutation_test",
         ck::asof_boundary_strict},
        {"window_frame_off_by_one", Hazard::kOperatorOrchestration,
         "tsx::mutant::Window{kFrameOffByOne}", "window_mutation_test",
         ck::window_frame_off_by_one},
        {"compress_decode_drift", Hazard::kOperatorOrchestration,
         "tsx::mutant::CompressedScan{kDropSecondDerivative}",
         "compress_mutation_test", ck::compress_decode_drift},
        {"parallel_merge_drop_partial", Hazard::kOperatorOrchestration,
         "mutant::ParallelEngine{kMergeDropPartial}", "wp10b_parallel_mutation_test",
         ck::parallel_merge_drop_partial},
        {"string_key_by_code", Hazard::kOperatorOrchestration,
         "mutant::StringKeyJoin{kHashRawCode}", "wp7b_strings_mutation_test",
         ck::string_key_by_code},
    };
}

}  // namespace

const std::vector<Entry>& catalog() {
    static const std::vector<Entry> kCatalog = build_catalog();
    return kCatalog;
}

}  // namespace qe::catalog
