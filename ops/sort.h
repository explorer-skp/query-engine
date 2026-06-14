//  WP-7: Sort (ORDER BY) — the pipeline-breaking operator that reorders its
//  child's rows by an ordered list of sort keys. Conforms to the frozen Operator
//  contract (ops/operator.h); this header is WP-7's OWN public surface (consumed
//  later by WP-8's dataframe builder), kept small and explicit: an ordered list of
//  (column, direction, null-ordering) keys.
//
//  MODEL (decision D3 pull-based; this operator is pipeline-breaking):
//   * open()  — open the child, DRAIN it fully into a dense materialized copy of
//     every output column, compute a sorted ROW PERMUTATION, then prepare to emit.
//   * next()  — yield the sorted rows in dense batches of <= kOutBatch rows, in
//     sorted order, until exhausted. The output schema is the child's schema
//     unchanged (sort reorders rows, never columns/types).
//   * close() — release state; the child is closed at end-of-drain.
//
//  TWO INDEPENDENT SORT PATHS (decision D10):
//   * COMPARISON sort — the general path. A std::stable_sort of a row-index array
//     under a left-to-right tuple comparator; handles ANY key type and mix.
//   * RADIX fast-path — for keys that are ALL fixed-width integer/timestamp
//     (I32/I64/TS): a stable LSD byte-wise radix over order-preserving normalized
//     keys. This is where vectorization earns its keep in the pipeline (the
//     whole-row permutation GATHER runs through the Highway gather kernels).
//   The two paths are SEPARATE algorithms (not a templated body selected by a
//   flag): radix-result == comparison-result is a meaningful cross-check, and a
//   classic radix sign-bit bug shows up as a disagreement (see the WP report and
//   the mutation self-test). Path::kAuto picks radix iff every key is integer.
//
//  WHOLE-ROW MOVEMENT. Sort computes a permutation of row indices, then GATHERS
//  every output column by that permutation (simd/gather_kernels.h — vector path +
//  scalar twin; GatherPath selects which, so scalar==vector is an end-to-end
//  check). Out-of-place by construction (the §12 aliasing hazard cannot arise).
//
//  STABILITY. Both paths are STABLE on the child's row order: rows with an equal
//  key tuple keep their original relative order. Stability is observable (and
//  checked) against the independent reference oracle on partial-key sorts.
//
//  SEMANTICS (must match DuckDB):
//   * Per-key ASC/DESC and per-key NULLS FIRST/LAST (null placement is absolute —
//     independent of ASC/DESC, per the SQL standard).
//   * Multi-column precedence is left-to-right (keys[0] is primary).
//   * F64 uses IEEE ordering; -0.0 == +0.0 and NaN are documented in the WP
//     report (the oracle generators do not emit them, keeping positional compare
//     unambiguous).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/operator.h"

namespace qe {

enum class SortDir { Asc, Desc };
enum class NullOrder { First, Last };

// One ORDER BY key: the child-output column index to sort on, its direction, and
// where NULLs land. Defaults mirror plain "ORDER BY c" (ASC NULLS LAST).
struct SortKey {
    std::uint32_t col = 0;
    SortDir dir = SortDir::Asc;
    NullOrder nulls = NullOrder::Last;
};

class Sort : public Operator {
   public:
    // Largest output batch (rows per next()); a tail batch may be smaller.
    static constexpr std::size_t kOutBatch = 2048;

    // Which sort algorithm to use. kAuto (production default) picks the radix
    // fast-path when every key column is integer-width (I32/I64/TS), else the
    // comparison path. The explicit values are a TEST SEAM so radix==comparison
    // can be driven on identical inputs.
    enum class Path { kAuto, kComparison, kRadix };

    // Which gather kernel moves whole rows by the permutation. kVector
    // (production default) uses the Highway gather; kScalar uses the twin. A TEST
    // SEAM so scalar==vector is an end-to-end check. (BOOL columns are 1 byte and
    // always use the scalar gather — documented in simd/gather_kernels.h.)
    enum class GatherPath { kVector, kScalar };

    // `keys`: the ORDER BY keys in precedence order (>=1). Takes ownership of the
    // child.
    Sort(std::unique_ptr<Operator> child, std::vector<SortKey> keys);

    // Test seams (call BEFORE open()). WP-8 ignores these (defaults are correct).
    void set_path(Path p) { path_ = p; }
    void set_gather_path(GatherPath g) { gather_path_ = g; }

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    std::unique_ptr<Operator> child_;
    std::vector<SortKey> keys_;
    Schema child_schema_;
    Path path_ = Path::kAuto;
    GatherPath gather_path_ = GatherPath::kVector;

    // Built during open(); opaque to keep the header light (mirrors Aggregate).
    struct State;
    std::shared_ptr<State> state_;
    std::size_t emit_cursor_ = 0;  // next sorted-row index to emit
    OwnedBatch current_;           // backs the view returned by next()
    bool opened_ = false;
};

}  // namespace qe
