//  WP-10b mutation self-test support (RIGOR.md rule 4 — "a checker that cannot fail
//  proves nothing"). This is a DELIBERATELY-BROKEN parallel driver: a planted
//  parallelism bug that the differential MUST catch while the real driver passes.
//
//  THE MUTANT (qe::mutant::ParallelMutation::kMergeDropPartial): the cross-WORKER
//  partial-aggregate merge OVERWRITES instead of ACCUMULATES — for a group key that
//  appears in two or more worker partials it keeps only the last partial and DROPS
//  the earlier ones (the brief's canonical "partial-aggregate merge that drops one
//  partial"). It is a bug that ONLY appears with >1 worker: the merge hooks fire
//  solely when a group is split across worker partials, so at 1 thread the result is
//  identical to the clean driver, and at 2/4/8 threads any group whose rows land in
//  more than one worker diverges (under-counted SUM/COUNT, wrong AVG). The
//  differential (parallel vs reference + DuckDB, and parallel vs single-thread)
//  catches it; the real driver is green.
//
//  Distinctly-named enum/lib (qe::mutant::ParallelMutation), separate code from the
//  real qe::exec::ParallelEngine — it only DERIVES and overrides the three protected
//  merge hooks. Hazard: kOperatorOrchestration (no new Hazard enumerator).
#pragma once

#include "exec/parallel.h"
#include "oracle/result_set.h"
#include "plan/plan.h"

namespace qe::mutant {

enum class ParallelMutation {
    kNone,
    kMergeDropPartial,  // cross-worker merge overwrites instead of accumulating
};

// Run `p` through a parallel driver carrying `mut`. With kNone this equals the
// clean qe::exec::run_plan_parallel; with kMergeDropPartial it plants the bug above.
qe::oracle::ResultSet run_plan_parallel_mutant(const qe::plan::Plan& p,
                                               qe::exec::ParallelConfig cfg,
                                               ParallelMutation mut);

}  // namespace qe::mutant
