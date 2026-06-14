//  WP-5: Hash aggregation / GROUP BY — the pipeline-breaking operator that
//  groups its child's rows by zero or more key columns and computes one or more
//  aggregates per group. Conforms to the frozen Operator contract
//  (ops/operator.h); this header is WP-5's OWN public surface (consumed later by
//  WP-8's dataframe builder), designed to be small and explicit.
//
//  MODEL (decision D3 pull-based; this operator is pipeline-breaking):
//   * open()  — open the child, DRAIN it fully, building the group table
//     (ops/hashtable.h, NullPolicy::kEqual) and accumulating per-group aggregate
//     state, then prepare to emit.
//   * next()  — yield the grouped result in dense batches of <= kOutBatch rows
//     (one row per group), keys first then aggregate columns, until exhausted.
//   * close() — release state; the child is closed at end-of-drain.
//
//  GROUPING.  `key_cols` lists the child-output column indices that form the
//  GROUP BY key, in key order. Zero keys => a single GLOBAL aggregate (exactly
//  one output row, even over empty input). One or more keys => GROUP BY, single
//  or composite; NULL keys group together (SQL semantics, NullPolicy::kEqual).
//
//  AGGREGATES (Phase 1): COUNT(*), COUNT(col), SUM, MIN, MAX, AVG. Multiple
//  aggregates are computed in one pass over the input.
//
//  NULL / EMPTY SEMANTICS (must match DuckDB):
//   * COUNT(*) counts ALL rows in the group; COUNT(col) counts NON-NULL values.
//   * SUM/MIN/MAX/AVG ignore NULL inputs. A group with zero non-null inputs
//     yields NULL (not 0). AVG = sum(non-null)/count(non-null), DOUBLE, NULL if
//     no non-null.
//   * Global aggregate over EMPTY input => one row: COUNT(*)=0, COUNT(col)=0,
//     SUM/MIN/MAX/AVG = NULL. GROUP BY over empty input => zero rows.
//
//  RESULT TYPE / OVERFLOW POLICY (documented; the oracle is made honest about
//  it — see the WP report and oracle/sql_render.cpp):
//   * COUNT(*) / COUNT(col)            -> I64
//   * SUM(I32) -> I64,  SUM(I64) -> I64,  SUM(F64) -> F64
//   * AVG(numeric)                     -> F64
//   * MIN(T) / MAX(T)                  -> T  (same type as the input column)
//   The SUM accumulator is I64 for integer inputs. DuckDB's SUM(INTEGER)/
//   SUM(BIGINT) return HUGEINT (128-bit) and never overflow; we bound the oracle
//   generators so every per-group sum provably fits I64 AND cast DuckDB's result
//   to BIGINT in the rendered SQL, so the differential is exact, never a silent
//   I64-vs-HUGEINT disagreement.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/hashtable.h"  // HashPath
#include "ops/operator.h"

namespace qe {

// The aggregate functions of Phase 1. CountStar is COUNT(*) (no input column);
// the rest read one input column.
enum class AggFunc { CountStar, Count, Sum, Min, Max, Avg };

// One aggregate to compute. `input_col` is the child-output column index the
// aggregate reads (ignored for CountStar). `out_name` is the published column
// name. Build via the factories so call sites read like the SQL they mirror.
struct AggSpec {
    AggFunc func = AggFunc::CountStar;
    std::uint32_t input_col = 0;  // ignored for CountStar
    std::string out_name;

    static AggSpec count_star(std::string name) {
        return {AggFunc::CountStar, 0, std::move(name)};
    }
    static AggSpec count(std::uint32_t col, std::string name) {
        return {AggFunc::Count, col, std::move(name)};
    }
    static AggSpec sum(std::uint32_t col, std::string name) {
        return {AggFunc::Sum, col, std::move(name)};
    }
    static AggSpec min(std::uint32_t col, std::string name) {
        return {AggFunc::Min, col, std::move(name)};
    }
    static AggSpec max(std::uint32_t col, std::string name) {
        return {AggFunc::Max, col, std::move(name)};
    }
    static AggSpec avg(std::uint32_t col, std::string name) {
        return {AggFunc::Avg, col, std::move(name)};
    }
};

// The result Type of `func` applied to an input column of type `input` (input is
// ignored for CountStar/Count). Single source of truth for the overflow policy
// above; used by output_schema(), the reference oracle, and the SQL renderer.
Type agg_result_type(AggFunc func, Type input);

// Which kernel path the GLOBAL (zero-key) reduction takes. Production uses
// kVector; the scalar==vector differential drives both and asserts identical
// results (RIGOR.md rule 3 / D17). Grouped (>=1 key) accumulation is an
// inherently-sequential scatter and ignores this (see the WP report).
enum class AggKernelPath { kVector, kScalar };

class Aggregate : public Operator {
   public:
    // Largest output batch (groups per next()); a tail batch may be smaller.
    static constexpr std::size_t kOutBatch = 2048;

    // `key_cols`: child-output column indices forming the GROUP BY key, in key
    // order (empty => global aggregate). `aggs`: the aggregates to compute
    // (>=1). Takes ownership of the child.
    Aggregate(std::unique_ptr<Operator> child, std::vector<std::uint32_t> key_cols,
              std::vector<AggSpec> aggs);

    // Test seam (call BEFORE open()): force the grouping hash path and the
    // global-reduction kernel path. Defaults are kVector/kVector; WP-8 ignores
    // this. Exposed so scalar==vector is a true end-to-end check.
    void set_paths(HashPath hash_path, AggKernelPath kernel_path) {
        hash_path_ = hash_path;
        kernel_path_ = kernel_path;
    }

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void drain_and_build();  // pull the child dry, build groups + state

    std::unique_ptr<Operator> child_;
    std::vector<std::uint32_t> key_cols_;
    std::vector<AggSpec> aggs_;
    Schema child_schema_;
    HashPath hash_path_ = HashPath::kVector;
    AggKernelPath kernel_path_ = AggKernelPath::kVector;

    // Built during open()/drain. The pImpl-free state lives in the .cpp via these
    // opaque members kept here for simplicity.
    struct State;
    std::shared_ptr<State> state_;  // shared_ptr so the header needs no full type
    std::size_t emit_cursor_ = 0;   // next group id to emit
    bool emitted_empty_global_ = false;  // global-over-empty: one row, once
    OwnedBatch current_;            // backs the view returned by next()
    bool opened_ = false;
};

}  // namespace qe
