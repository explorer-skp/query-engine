//  WP-5: TEST-ONLY planted-mutant aggregation operator. "A checker that cannot
//  fail proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::Aggregate is a faithful copy of the real GROUPED accumulation /
//  finalization / output-batching loops (ops/aggregate.cpp) that REUSES the exact
//  shared helpers (ops/agg_internal.h) — so the ONLY difference from the real
//  operator is one deliberately-broken step, selected by the `Mutation` enum. The
//  mutation self-test (tests/agg_mutation_test.cpp) runs the real operator and a
//  mutant through the SAME differential (reference + DuckDB) and shows the diff
//  CATCHES each mutant, while the real operator passes.
//
//  Planted mutants (all in the GROUP BY path; the test drives keyed queries):
//   * kFoldNullInSum     — the §5 named bug: SUM/AVG fold a NULL input as 0 AND
//     count it, instead of ignoring it. An all-null group then emits 0 (and a
//     non-NULL "seen") instead of NULL; COUNT/AVG denominators are wrong.
//   * kEmptyGroupZero    — finalize emits the accumulator even for a group with
//     zero non-null inputs (emits 0 / identity instead of NULL).
//   * kGroupTailOffByOne — output batching drops the LAST group (off-by-one on
//     the group range), so the final partial output batch is short a row.
//
//  This is NOT linked into any engine target — only the mutation self-test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/aggregate.h"   // AggSpec
#include "ops/operator.h"

namespace qe::mutant {

enum class Mutation {
    kFoldNullInSum,
    kEmptyGroupZero,
    kGroupTailOffByOne,
    kMaxDropsNan  // F64 MAX via raw std::max: NaN inputs silently dropped (audit C2)
};

// Mirror of qe::Aggregate's GROUP BY surface, with a planted defect. Requires at
// least one key column (the mutants live in the grouped path).
class Aggregate : public Operator {
   public:
    static constexpr std::size_t kOutBatch = 2048;

    Aggregate(std::unique_ptr<Operator> child,
              std::vector<std::uint32_t> key_cols, std::vector<AggSpec> aggs,
              Mutation mut);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void drain_and_build();

    std::unique_ptr<Operator> child_;
    std::vector<std::uint32_t> key_cols_;
    std::vector<AggSpec> aggs_;
    Mutation mut_;
    Schema child_schema_;

    struct State;
    std::shared_ptr<State> state_;
    std::size_t emit_cursor_ = 0;
    OwnedBatch current_;
    bool opened_ = false;
};

}  // namespace qe::mutant
