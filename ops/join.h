//  WP-6: Hash equi-join — the pipeline operator that joins a PROBE input against
//  a BUILD input on one or more equi-keys. Conforms to the frozen Operator
//  contract (ops/operator.h); this header is WP-6's OWN public surface (consumed
//  later by WP-8's dataframe builder), designed to be small and explicit — it
//  mirrors the AggSpec-style shape of ops/aggregate.h.
//
//  MODEL (decision D3 pull-based; the BUILD side is pipeline-breaking):
//   * open()  — open the BUILD child, DRAIN it fully into the frozen HashTable
//     (ops/hashtable.h, NullPolicy::kNeverMatch — SQL join NULLs never match),
//     materialize the build-side columns, and build the multiplicity map
//     group_id -> [build row indices] (the table dedups distinct keys to ONE
//     group id and stores no payloads, so collecting the matching rows per group
//     is THIS operator's job). Then open the PROBE child.
//   * next()  — pull PROBE batches; HashTable::find() maps each probe row to a
//     group id (or kNoGroup). For each match emit one output row per build row in
//     that group's list (gather build columns + carry probe columns). Output is
//     produced in dense <= kOutBatch-row batches; a single high-fanout probe row
//     (or LEFT-join unmatched row) is split across batches at the row granularity.
//   * close() — release state; both children are closed at end-of-drain / probe
//     exhaustion.
//
//  OUTPUT SCHEMA / COLUMN ORDER (documented contract): ALL probe columns in their
//  child order, FOLLOWED BY all build columns in their child order. Field names
//  are the children's names verbatim (probe then build) and MAY collide across
//  sides — consumers identify columns by POSITION, not name (the WP-6 report and
//  the SQL renderer alias by position). Types are the probe column types followed
//  by the build column types.
//
//  JOIN VARIANTS:
//   * Inner — emit only matching (probe,build) pairs; a probe row with k build
//     matches emits k rows; unmatched probe rows emit nothing.
//   * Left  — every probe row appears at least once; an unmatched probe row emits
//     exactly one row with ALL build columns NULL.
//
//  NULL / FLOAT KEY SEMANTICS (must match DuckDB; validated by the oracle):
//   * A key tuple containing ANY NULL matches nothing — not even an identical NULL
//     tuple (NullPolicy::kNeverMatch). A NULL key on EITHER side joins nothing.
//   * F64 key equality uses the HashTable's canonicalization (-0.0 == +0.0,
//     NaN == NaN), matching DuckDB's join-key semantics rather than raw IEEE `==`.
//   * Composite keys match iff ALL key columns match (positionally, in key order).
//     The probe key column types must equal the build key column types, in order.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/hashtable.h"  // HashPath
#include "ops/operator.h"

namespace qe {

// INNER keeps only matching pairs; LEFT preserves every probe (left) row,
// NULL-filling the build columns when a probe row has no match.
enum class JoinType { Inner, Left };

// Which code path performs the build-column GATHER during emit. Production uses
// kVector (Highway GatherIndex, simd/gather_kernels.h); the scalar==vector
// differential drives both and asserts identical output (RIGOR.md rule 3 / D17).
// BOOL columns have no vector gather twin and always use the scalar byte gather
// regardless of this setting (documented, not an omission — see simd/gather_kernels.h).
enum class GatherPath { kVector, kScalar };

class HashJoin : public Operator {
   public:
    // Largest output batch (rows per next()); a tail batch may be smaller.
    static constexpr std::size_t kOutBatch = 2048;

    // `probe`/`build` are the two child operators (this takes ownership of both).
    // `probe_keys` / `build_keys` are the equi-key column indices into each
    // child's output schema, in key order; the two lists must have equal length
    // and positionally-matching types (single key => length 1; composite =>
    // length > 1). `type` selects INNER vs LEFT-OUTER.
    HashJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
             std::vector<std::uint32_t> probe_keys,
             std::vector<std::uint32_t> build_keys, JoinType type);

    // Test seam (call BEFORE open()): force the probe hash path and the
    // build-column gather path. Defaults are kVector/kVector; WP-8 ignores this.
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
    void build_side();                 // drain build child, fill table + store
    bool build_pairs_for_probe();      // pull one probe batch; fill pair arrays.
                                       // returns false at probe exhaustion.

    std::unique_ptr<Operator> probe_;
    std::unique_ptr<Operator> build_;
    std::vector<std::uint32_t> probe_keys_;
    std::vector<std::uint32_t> build_keys_;
    JoinType type_;
    Schema probe_schema_;
    Schema build_schema_;
    HashPath hash_path_ = HashPath::kVector;
    GatherPath gather_path_ = GatherPath::kVector;

    // Built during open(); opaque state lives in the .cpp (mirrors Aggregate).
    struct State;
    std::shared_ptr<State> state_;

    // Probe-side streaming cursor. cur_probe_ holds the live probe batch whose
    // pairs we are draining; its Column views stay valid until the next
    // probe_->next() (which we only call once pairs are exhausted).
    std::optional<Batch> cur_probe_;
    std::size_t pair_cursor_ = 0;  // next pair index to emit from the arrays
    OwnedBatch current_;           // backs the view returned by next()
    bool opened_ = false;
    bool probe_done_ = false;
};

}  // namespace qe
