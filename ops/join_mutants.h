//  WP-6: TEST-ONLY planted-mutant hash-join operator. "A checker that cannot fail
//  proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::HashJoin is a faithful copy of the real join's build / pair-
//  generation / output-batching loops (ops/join.cpp) that REUSES the exact shared
//  helpers (ops/join_internal.h: BuildStore + the gather emitters) — so the ONLY
//  difference from the real operator is one deliberately-broken step, selected by
//  the `Mutation` enum. The mutation self-test (tests/join_mutation_test.cpp) runs
//  the real operator and a mutant through the SAME differential (reference +
//  DuckDB) and shows the diff CATCHES each mutant, while the real operator passes.
//
//  Planted mutants:
//   * kDropProbeMatch       — the §5 named bug: skip emitting ONE matched output
//     row (the first matched pair), so the output is short a row.
//   * kLeftWrongNull        — a LEFT-join unmatched probe row emits build row 0's
//     real (valid) values instead of NULLs.
//   * kCompositeFirstKeyOnly— build/probe the table over only the FIRST key
//     column, so a composite-key join matches on the first key alone (over-joins).
//   * kFanoutTailOffByOne   — the final output batch is emitted one row short
//     (off-by-one on the chunk tail) on a high-fanout probe batch.
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
#include "ops/join.h"  // JoinType, GatherPath
#include "ops/operator.h"

namespace qe::mutant {

enum class Mutation {
    kDropProbeMatch,
    kLeftWrongNull,
    kCompositeFirstKeyOnly,
    kFanoutTailOffByOne,
};

// Mirror of qe::HashJoin with a planted defect.
class HashJoin : public Operator {
   public:
    static constexpr std::size_t kOutBatch = 2048;

    HashJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
             std::vector<std::uint32_t> probe_keys,
             std::vector<std::uint32_t> build_keys, JoinType type, Mutation mut);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void build_side();
    bool build_pairs_for_probe();

    std::unique_ptr<Operator> probe_;
    std::unique_ptr<Operator> build_;
    std::vector<std::uint32_t> probe_keys_;
    std::vector<std::uint32_t> build_keys_;
    JoinType type_;
    Mutation mut_;
    Schema probe_schema_;
    Schema build_schema_;

    struct State;
    std::shared_ptr<State> state_;
    std::optional<Batch> cur_probe_;
    std::size_t pair_cursor_ = 0;
    OwnedBatch current_;
    bool opened_ = false;
    bool probe_done_ = false;
    bool dropped_one_ = false;  // kDropProbeMatch: drop exactly one match
};

}  // namespace qe::mutant
