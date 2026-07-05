//  WP-9: declarations of every mutation-catalog check, defined across the
//  conflict-free check TUs (tests/catalog_checks_*.cpp). The registry
//  (tests/mutation_catalog.cpp) references these without pulling in any mutant
//  header, so the three distinct qe::mutant::Mutation enums never collide in one
//  TU. Each returns a Verdict reproducing the decisive clean-vs-mutant comparison.
#pragma once

#include "tests/mutation_catalog.h"

namespace qe::catalog::checks {

// kernel + new-mutant checks (tests/catalog_checks_kernel.cpp)
Verdict simd_tail();
Verdict null_propagation();
Verdict all_valid_fastpath();
Verdict selection_aliasing();
Verdict float_to_int_round();
Verdict aggregation_overflow();

// hash table (tests/catalog_checks_hash.cpp)
Verdict hash_probe_overflow();
Verdict hash_growth_rehash();

// join + sort (tests/catalog_checks_join_sort.cpp)
Verdict join_drop_match();
Verdict sort_desc_as_asc();

// aggregate + plan (tests/catalog_checks_agg_plan.cpp)
Verdict agg_fold_null_in_sum();
Verdict agg_max_drops_nan();
Verdict plan_drop_sort();

// as-of join (tests/catalog_checks_asof.cpp) — WP-12
Verdict asof_boundary_strict();

// windowed aggregation (tests/catalog_checks_window.cpp) — WP-13
Verdict window_frame_off_by_one();

// compressed scan (tests/catalog_checks_compress.cpp) — WP-14
Verdict compress_decode_drift();

// parallel execution (tests/catalog_checks_parallel.cpp) — WP-10b
Verdict parallel_merge_drop_partial();

// dictionary-encoded VARCHAR join keys (tests/catalog_checks_strings.cpp) — WP-7b
Verdict string_key_by_code();

}  // namespace qe::catalog::checks
