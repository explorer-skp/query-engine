//  WP-12: BACKWARD AS-OF JOIN — the Phase-2 headline tick-data operator. For each
//  PROBE (left) row with partition-key tuple k and timestamp t, it finds the BUILD
//  (right) row with the SAME key k whose timestamp tb is the GREATEST tb <= t
//  (nearest PRECEDING — "backward as-of") and emits the probe columns followed by
//  that build row's columns. Conforms to the frozen Operator contract
//  (ops/operator.h); THIS header is WP-12's OWN public surface (frozen at
//  acceptance — WP-13/WP-15 may reference it), kept small and explicit, mirroring
//  the AggSpec/HashJoin shape of ops/join.h.
//
//  SEMANTICS (verified against DuckDB v1.1.3 `ASOF JOIN`; see the WP report):
//   * BOUNDARY is `>=`, not `>`: a build row whose timestamp EQUALS the probe
//     timestamp DOES match (it is the greatest tb <= t). Using `>` (dropping
//     equal-timestamp matches) is the classic as-of bug and a mutation self-test.
//   * INNER drops a probe row with no preceding build match; LEFT emits it once
//     with ALL build columns NULL.
//   * NULL never matches: a probe/build row with a NULL in ANY key column, or a
//     NULL timestamp, participates in no match (principled SQL semantics: `=`/`>=`
//     on a NULL operand is NULL => non-match). An unmatched probe row follows the
//     INNER / LEFT rule above. NULL KEYS agree with DuckDB. NULL TIMESTAMPS are a
//     DOCUMENTED DIVERGENCE: DuckDB v1.1.3's ASOF instead matches a NULL probe
//     timestamp to a NULL build timestamp within the same key partition; we take
//     the principled never-match path and keep NULL timestamps out of the
//     differential grammar (the oracle generators do not emit them — see the WP
//     report), validating the engine's NULL-timestamp behavior against the
//     independent reference directly.
//   * TIES: if the build side has duplicate (key, timestamp) rows the chosen match
//     is ambiguous (DuckDB does not define it either). The oracle generators keep
//     build-side (key, timestamp) UNIQUE so the differential is deterministic — see
//     the WP report; this operator does not itself promise a tie-break.
//
//  OPTIONAL "WITHIN TOLERANCE" VARIANT (decision: pandas `merge_asof(tolerance=)` /
//  kdb `aj` window semantics): when `tolerance` is set, the nearest PRECEDING build
//  row is found FIRST and then DROPPED if t - tb > tolerance (so the row is treated
//  as no-match: INNER drops, LEFT NULL-fills). Because the nearest-preceding tb is
//  the closest from below, "nearest then check" and "nearest within the window"
//  coincide for a backward as-of. `tolerance` is in the SAME integer units as the
//  timestamp column (ns for TS). See the WP report for the exact DuckDB rendering.
//
//  APPROACH (D10 / §5 — sorted-merge-per-key-group; reuses the frozen pieces, does
//  NOT reimplement them):
//   * SORT both inputs by (key columns..., timestamp) ascending via the frozen Sort
//     operator (ops/sort.h). Build groups then arrive timestamp-ascending and probe
//     rows of one key form a single contiguous, timestamp-ascending run.
//   * PARTITION by key via the frozen HashTable (ops/hashtable.h,
//     NullPolicy::kNeverMatch — a NULL key never matches): each build row gets a
//     stable group id and each group keeps its build rows in timestamp order; a
//     probe row's group id is found WITHOUT replicating the sort's key comparator.
//   * MERGE per key group with a PER-KEY ADVANCING CURSOR: walking a group's probe
//     rows in ascending timestamp, a build cursor only moves forward to the last
//     tb <= t (O(n+m) per group). This merge is sequential scalar control flow (the
//     WP-6 probe-walk / WP-5 grouped-scatter precedent — fine to be scalar).
//   * MATERIALIZE the matched build row + carried probe row through the WP-1 gather
//     kernels (simd/gather_kernels.h) via the shared emitters (ops/join_internal.h):
//     vector path + scalar twin, selected by GatherPath, so scalar==vector is an
//     end-to-end check. Output is produced in dense <= kOutBatch-row batches.
//
//  OUTPUT SCHEMA / COLUMN ORDER (documented contract, same shape as HashJoin): ALL
//  probe columns in their child order, FOLLOWED BY all build columns in their child
//  order. Field names are the children's verbatim and MAY collide across sides —
//  consumers identify columns by POSITION, not name. Types are the probe column
//  types followed by the build column types.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/hashtable.h"  // HashPath, NullPolicy
#include "ops/join.h"       // GatherPath (reused; same vector/scalar gather seam)
#include "ops/operator.h"

namespace qe::tsx {

// INNER drops unmatched probe rows; LEFT preserves every probe row, NULL-filling
// the build columns when a probe row has no preceding match (or is out of
// tolerance / has a NULL key or timestamp).
enum class AsofType { Inner, Left };

class AsofJoin : public Operator {
   public:
    // Largest output batch (rows per next()); a tail batch may be smaller.
    static constexpr std::size_t kOutBatch = 2048;

    // `probe`/`build` are the two child operators (this takes ownership of both).
    // `probe_keys` / `build_keys` are the EQUALITY partition-key column indices
    // into each child's output schema, in key order (0 or more; equal length;
    // positionally-matching types — composite keys match iff ALL match). An EMPTY
    // key list means a single global ordering (every row shares one partition).
    // `probe_time` / `build_time` are the single ORDERING (timestamp) column index
    // per side (Type TS or an integer type I32/I64; the two time column types
    // should be comparable — read as int64). `type` selects INNER vs LEFT. When
    // `tolerance` is set, a match is kept only if t - tb <= *tolerance (in the
    // timestamp column's integer units); std::nullopt means unbounded.
    AsofJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
             std::vector<std::uint32_t> probe_keys,
             std::vector<std::uint32_t> build_keys, std::uint32_t probe_time,
             std::uint32_t build_time, AsofType type,
             std::optional<std::int64_t> tolerance = std::nullopt);

    // Test seam (call BEFORE open()): force the probe/build hash path and the
    // column gather path. Defaults are kVector/kVector; downstream WPs ignore this.
    // Exposed so scalar==vector is a true end-to-end check.
    void set_paths(HashPath hash_path, GatherPath gather_path) {
        hash_path_ = hash_path;
        gather_path_ = gather_path;
    }

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void build_side();             // drain sorted build child; fill table + store
    bool build_pairs_for_probe();  // pull one sorted probe batch; fill pair arrays.
                                   // returns false at probe exhaustion.

    std::unique_ptr<Operator> probe_;
    std::unique_ptr<Operator> build_;
    std::vector<std::uint32_t> probe_keys_;
    std::vector<std::uint32_t> build_keys_;
    std::uint32_t probe_time_;
    std::uint32_t build_time_;
    AsofType type_;
    std::optional<std::int64_t> tolerance_;
    Schema probe_schema_;
    Schema build_schema_;
    HashPath hash_path_ = HashPath::kVector;
    GatherPath gather_path_ = GatherPath::kVector;

    // Built during open(); opaque state lives in the .cpp (mirrors HashJoin/Sort).
    struct State;
    std::shared_ptr<State> state_;

    // Probe-side streaming cursor over the SORTED probe stream. cur_probe_ holds
    // the live sorted probe batch whose pairs we are draining; its Column views
    // stay valid until the next probe_->next() (pulled only once pairs exhausted).
    std::optional<Batch> cur_probe_;
    std::size_t pair_cursor_ = 0;  // next pair index to emit from the arrays
    OwnedBatch current_;           // backs the view returned by next()
    bool opened_ = false;
    bool probe_done_ = false;
};

}  // namespace qe::tsx
