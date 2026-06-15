//  WP-8 MUTATION SELF-TEST support (RIGOR.md rule 4: "a checker that cannot fail
//  proves nothing"). lower_mutant() is a DELIBERATELY-WRONG transcription of
//  Plan::lower() — it commits one plan-LOWERING defect of the brief's catalog —
//  so the plan differential (vs the independent reference AND DuckDB) can be SHOWN
//  to flag it, then pass on the correct Plan::lower(). It is test-only and is
//  NEVER linked into any engine target (only the WP-8 mutation self-test).
#pragma once

#include <cstddef>
#include <memory>

#include "ops/operator.h"
#include "ops/scan.h"  // Scan::kDefaultBatchSize
#include "plan/plan.h"

namespace qe::plan {

// One planted lowering defect:
//   kDropFilter      — drop the Filter node (lower its child directly): keeps rows
//                      the predicate rejects => row-count / value mismatch.
//   kSwapJoinSides   — swap a join's probe/build children (and their keys): output
//                      column order/types reorder (build cols first) => mismatch.
//   kReverseAggKeys  — lower GROUP BY keys in reversed order: key output columns
//                      reorder => mismatch on composite keys.
//   kDropSort        — drop the Sort node: rows emitted unsorted => POSITIONAL
//                      (ORDERED) compare mismatch.
enum class LowerMutation { kDropFilter, kSwapJoinSides, kReverseAggKeys, kDropSort };

// Lower `p` to an operator tree applying `m` at EVERY node of the matching kind.
// Identical to Plan::lower() except for the planted defect.
std::unique_ptr<Operator> lower_mutant(
    const Plan& p, LowerMutation m,
    std::size_t batch_size = Scan::kDefaultBatchSize);

}  // namespace qe::plan
