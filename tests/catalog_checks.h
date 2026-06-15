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
Verdict plan_drop_sort();

}  // namespace qe::catalog::checks
