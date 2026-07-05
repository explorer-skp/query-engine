//  WP-10b: the optional, ADDITIVE parallel-execution layer (decision D15). It takes
//  a built plan::Plan plus a thread count and runs it across worker threads,
//  returning the SAME logical result as plan.lower() drained single-threaded — the
//  oracle proves parallel == single-thread == reference == DuckDB (canonicalized for
//  unordered results, positional for a top-level Sort).
//
//  NO FROZEN-INTERFACE CHANGE. The frozen pull-based Operator (ops/operator.h) and
//  every operator/plan/oracle surface are unchanged; this layer COMPOSES them. It
//  lives in its own qe::exec namespace and dir.
//
//  MORSEL-DRIVEN MODEL (Leis et al., scoped to what the oracle can verify):
//   * MORSEL = a 64-aligned contiguous row range of the DRIVING scan (the bottom-
//     left base table of the pipeline). Worker threads pull morsels from a shared
//     ATOMIC cursor (a shared dispenser; work-stealing not required).
//   * PER-PIPELINE INSTANCES: each morsel is executed through a PRIVATE operator
//     tree (exec/morsel_scan.h leaf + the frozen operators), so no engine state is
//     shared between threads. Only the atomic cursor is shared-mutable; every other
//     mutable structure is worker-private and combined AFTER the threads join
//     (a happens-before barrier) — that is the whole race-freedom argument.
//   * EXCHANGE / MERGE, by the plan's top operator:
//       - Scan/Filter/Project/Join (streaming): each morsel emits partial output;
//         the exchange CONCATENATES them (order-free; the diff canonicalizes, D12).
//         A Join parallelizes the PROBE morsels against a build-side hash table that
//         each MORSEL's private tree rebuilds it (a full build per morsel, not
//         merely per worker — more redundant than the brief's concession but
//         correct; the shared-build-once seam remains future work — acceptable
//         per the WP brief; the shared-build optimization is future work).
//       - Aggregate (group-by): each WORKER accumulates a PRIVATE partial hash
//         aggregate over all its morsels (AVG decomposed to SUM+COUNT so partials
//         are mergeable); the cross-worker MERGE combines partials per group key
//         (sum the sums/counts, min the mins, max the maxes; AVG = merged sum /
//         merged count). Global (zero-key) aggregate merges to one row.
//       - Sort (ORDER BY): the upstream pipeline feeding the Sort is parallelized;
//         the final Sort runs SINGLE-THREAD over the merged input (the WP brief's
//         sanctioned simpler choice — documented). Order is preserved for the
//         positional diff.
//   * Operators outside this set (AsofJoin/Window/CompressedScan, or a nested
//     pipeline-breaker on the driving path) are NOT morsel-parallelized in scope:
//     supported() returns false and run() falls back to a correct single-thread
//     execution (and SAYS SO — see the WP report).
//
//  THREAD COUNT is configuration, never hardcoded (RIGOR.md rule 2): it is a
//  parameter, defaulting to std::thread::hardware_concurrency(). No width / ISA /
//  cache / core constant appears in any signature.
#pragma once

#include <cstddef>
#include <cstdint>

#include "oracle/result_set.h"
#include "plan/plan.h"

namespace qe::exec {

// Default morsel granularity (rows). A multiple of 64 (validity-word alignment).
// Configurable; intentionally smaller than a full table so several morsels exist
// per thread (the dispenser is the point) and so group keys span morsels/workers.
inline constexpr std::size_t kDefaultMorselRows = 1024;

struct ParallelConfig {
    // 0 => std::thread::hardware_concurrency() (at least 1). Never a baked count.
    unsigned threads = 0;
    // 0 => kDefaultMorselRows. Must be a positive multiple of 64 when nonzero.
    std::size_t morsel_rows = 0;
};

// The parallel driver. Public so the parallelism mutant (exec/parallel_mutants.h)
// can DERIVE from it and override ONLY the cross-worker merge hooks below — keeping
// the planted bug separate code from the real driver (RIGOR.md rule 4), and making
// it a bug that bites ONLY with >1 worker (the merge hooks fire only when a group
// is split across worker partials).
class ParallelEngine {
   public:
    explicit ParallelEngine(ParallelConfig cfg = {});
    virtual ~ParallelEngine() = default;

    // Run `p` and return its fully materialized result (same logical result as the
    // single-thread plan path). For a top-level Sort the row order is the sort
    // order (positional diff); otherwise the row set is order-free (D12).
    qe::oracle::ResultSet run(const qe::plan::Plan& p);

    // Is `p` in the morsel-parallelizable subset (Scan/Filter/Project/Join on the
    // driving path, optionally capped by one Aggregate and/or one top Sort)? When
    // false, run() executes single-threaded.
    static bool supported(const qe::plan::Plan& p);

   protected:
    // Cross-WORKER partial-aggregate merge hooks. Production = exact combine. The
    // mutant overrides these to DROP earlier partials (overwrite instead of
    // accumulate); since they fire only when a group key appears in >=2 worker
    // partials, the mutant is clean at 1 thread and diverges at >=2 (see the WP
    // report + tests/catalog_checks_parallel.cpp). MIN/MAX merge is idempotent and
    // not routed through a hook.
    virtual std::int64_t merge_count(std::int64_t acc, std::int64_t add) const {
        return acc + add;
    }
    virtual std::int64_t merge_sum_i(std::int64_t acc, std::int64_t add) const {
        return acc + add;
    }
    virtual double merge_sum_f(double acc, double add) const {
        return acc + add;
    }

   private:
    qe::oracle::ResultSet run_node(const qe::plan::Plan& p);
    qe::oracle::ResultSet run_streaming(const qe::plan::Plan& p);
    qe::oracle::ResultSet run_aggregate(const qe::plan::Plan& p);

    unsigned threads_;
    std::size_t morsel_rows_;
};

// Convenience: construct a (clean) ParallelEngine and run `p`.
qe::oracle::ResultSet run_plan_parallel(const qe::plan::Plan& p,
                                        ParallelConfig cfg = {});

}  // namespace qe::exec
