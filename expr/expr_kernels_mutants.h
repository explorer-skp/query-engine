//  WP-2: TEST-ONLY planted mutants for the expression mutation self-test.
//  "A checker that cannot fail proves nothing." Each carries one deliberate bug
//  drawn from the §12 expression hotspots; tests/expr_mutation_test.cpp shows the
//  suite CATCHES each, then that the real path passes the same check. NOT linked
//  into any engine target.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/column.h"
#include "core/owned_batch.h"

namespace qe::expr::mutant {

// HOTSPOT: SIMD tail/remainder. A comparison vector kernel (i32 <) that handles
// only the full-vector body and DROPS the scalar tail — wrong for any length
// that is not a whole number of vectors. Caught by scalar==vector at 63/65/2047.
void cmp_lt_i32_tailbug_vec(const std::int32_t* a, const std::int32_t* b,
                            std::uint8_t* out, std::size_t n);

// HOTSPOT: null propagation through three-valued logic. A degraded AND that uses
// plain "null-if-any" (NULL if either operand is NULL) instead of Kleene rules,
// so FALSE ∧ NULL is wrongly NULL instead of FALSE. Tri-state in/out (0=F,1=N,
// 2=T). Caught by the AND truth table.
void logic_and_twovalued(const std::uint8_t* a, const std::uint8_t* b,
                         std::uint8_t* out, std::size_t n);

// HOTSPOT: the all-valid fast path skipping a real null. A null-propagation
// combine that takes the all_valid shortcut when EITHER operand is all-valid
// (correct rule: only when BOTH are), dropping the other operand's nulls.
void propagate_nulls_and_allvalidbug(OwnedColumn& out, const Column& a,
                                     const Column& b);

}  // namespace qe::expr::mutant
