//  WP-13: WINDOWED / TIME-BUCKETED AGGREGATION — a Phase-2 tick-data operator. ONE
//  operator (qe::tsx::Window) carries BOTH window modes via WindowMode; it conforms
//  to the frozen Operator contract (ops/operator.h). THIS header is WP-13's OWN
//  public surface (frozen at acceptance), kept small and explicit, and it REUSES the
//  Phase-1 aggregate vocabulary VERBATIM (ops/aggregate.h: AggFunc / AggSpec /
//  agg_result_type) — there is no new aggregate vocabulary.
//
//  TWO MODES (both required; selected by WindowMode):
//
//  * TUMBLING (time-bucketed aggregate). Partition by zero or more equality key
//    columns PLUS a derived integer time bucket; aggregate per (keys…, bucket).
//      - bucket_id   = t / W            (integer floor division; W > 0)
//      - bucket_start = bucket_id * W   (the bucket's LOWER EDGE), emitted as the
//        bucket column, typed as the timestamp column's type (TS / I64 / I32).
//    Output schema: [partition keys…, bucket_start, aggregate columns…] — one row
//    per OCCUPIED (keys…, bucket). NULL partition keys group together (SQL GROUP BY
//    semantics, NullPolicy::kEqual), exactly like ops/aggregate.h. Output is
//    UNORDERED (the differential canonicalizes on all output columns, D12); float
//    aggregates use the D11 tolerance.
//      GENERATION CONSTRAINT (honesty lever, mirrors WP-12 "constrain the grammar"):
//      the differential generators emit timestamps >= 0, so integer bucketing is
//      divergence-free vs DuckDB (negative floor-division rounding is a known SQL/`/`
//      divergence we keep OUT of the grammar). This operator does not itself promise
//      a behavior for negative t (documented divergence).
//
//  * SLIDING (running aggregate over an N-preceding-rows frame). ROW-PRESERVING:
//    exactly one output row per input row. Frame = ROWS BETWEEN P PRECEDING AND
//    CURRENT ROW, partitioned by the equality keys, ordered by the timestamp
//    ascending; P >= 0 is a row count. Output schema: ALL child columns in order,
//    then one column per AggSpec (the running value at that row over its frame).
//    COUNT(*) over the frame counts frame rows; SUM/MIN/MAX/AVG/COUNT(col) follow the
//    SAME null rules as ops/aggregate.h (NULLs ignored; an all-null/empty frame slice
//    yields NULL for SUM/MIN/MAX/AVG, 0 for COUNT). Output is UNORDERED (each row
//    carries its full input tuple + running value, so rows are distinguishable; the
//    differential canonicalizes on all output columns).
//      GENERATION CONSTRAINT: per-partition timestamps are GLOBALLY DISTINCT (the
//      WP-12 determinism lever), so the window's ORDER BY t is a TOTAL order within
//      each partition => every row's running value is deterministic.
//
//  APPROACH (reuses the frozen pieces; does NOT reimplement them — see tsx/window.cpp):
//   * Tumbling: derive the integer bucket per row, then GROUP BY (partition keys…,
//     bucket) via the frozen HashTable (ops/hashtable.h, NullPolicy::kEqual) and
//     accumulate per-group aggregate state — the same shape as ops/aggregate.h.
//     Grouped accumulation is sequential scalar scatter (the WP-5 precedent:
//     conflict-free SIMD scatter is not portably expressible in Highway).
//   * Sliding: SORT the child by (keys…, t) ascending via the frozen Sort
//     (ops/sort.h) so each partition is one contiguous, time-ascending run; then a
//     per-partition advancing cursor (running sum/count for SUM/AVG/COUNT; a
//     monotonic deque for MIN/MAX) computes the running aggregate. Sequential scalar
//     control flow (the WP-6 probe-walk / WP-12 merge precedent).
//   * MATERIALIZE the sliding output's CHILD columns through the WP-1 gather kernels
//     (simd/gather_kernels.h) via the shared emitters (ops/join_internal.h:
//     BuildStore + emit_build_column), VECTOR path + scalar twin selected by
//     GatherPath — the one vectorized kernel with a scalar twin, so scalar==vector is
//     a true end-to-end check. The bucket-scatter and ring-buffer are scalar by the
//     precedent above. (Tumbling emits group keys read back from the table, like
//     ops/aggregate.h, so it has no gather; the GatherPath seam bites on sliding.)
//   * Output is produced in dense <= kOutBatch-row batches; a tail batch may be
//     smaller.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"  // AggFunc / AggSpec / agg_result_type (REUSED verbatim)
#include "ops/hashtable.h"  // HashPath, NullPolicy
#include "ops/join.h"       // GatherPath (reused; same vector/scalar gather seam)
#include "ops/operator.h"

namespace qe::tsx {

// The two window modes. Tumbling = time-bucketed aggregate (one row per occupied
// bucket); Sliding = running aggregate over an N-preceding-rows frame (one row per
// input row).
enum class WindowMode { Tumbling, Sliding };

class Window : public Operator {
   public:
    // Largest output batch (rows per next()); a tail batch may be smaller.
    static constexpr std::size_t kOutBatch = 2048;

    // `child` is the single input operator (this takes ownership). `keys` are the
    // EQUALITY partition-key column indices into the child's output schema, in key
    // order (0 or more; an empty key list means a single global partition). `time`
    // is the single ordering/bucketing timestamp column index (Type TS or an integer
    // type I32/I64; read as int64). `param` is the bucket WIDTH W (> 0) for Tumbling
    // or the PRECEDING row count P (>= 0) for Sliding. `aggs` are the aggregates to
    // compute (>= 1), reusing ops/aggregate.h AggSpec verbatim.
    Window(std::unique_ptr<Operator> child, WindowMode mode,
           std::vector<std::uint32_t> keys, std::uint32_t time,
           std::int64_t param, std::vector<AggSpec> aggs);

    // Test seam (call BEFORE open()): force the partition hash path (Tumbling) and
    // the output column gather path (Sliding). Defaults are kVector/kVector;
    // downstream consumers ignore this. Exposed so scalar==vector is a true
    // end-to-end check (RIGOR.md rule 3 / D17).
    void set_paths(HashPath hash_path, GatherPath gather_path) {
        hash_path_ = hash_path;
        gather_path_ = gather_path;
    }

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void build_tumbling();              // drain child; group by (keys, bucket)
    void build_sliding();               // sort child; compute running aggregates
    std::optional<Batch> next_tumbling();
    std::optional<Batch> next_sliding();

    std::unique_ptr<Operator> child_;
    WindowMode mode_;
    std::vector<std::uint32_t> keys_;
    std::uint32_t time_;
    std::int64_t param_;             // W (tumbling) / P (sliding)
    std::vector<AggSpec> aggs_;
    Schema child_schema_;
    HashPath hash_path_ = HashPath::kVector;
    GatherPath gather_path_ = GatherPath::kVector;

    // Built during open(); opaque state lives in the .cpp (mirrors Aggregate/Sort).
    struct State;
    std::shared_ptr<State> state_;
    std::size_t emit_cursor_ = 0;  // next group / sorted-row index to emit
    OwnedBatch current_;           // backs the view returned by next()
    bool opened_ = false;
};

}  // namespace qe::tsx
