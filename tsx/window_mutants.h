//  WP-13: TEST-ONLY planted-mutant windowed-aggregation operator. "A checker that
//  cannot fail proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::Window is a faithful copy of the real Window operator's tumbling
//  group/scatter loop and sliding sort/sweep/emit loops (tsx/window.cpp) that REUSES
//  the exact shared pieces — the ops/join_internal.h BuildStore + emit_build_column
//  gather, the ops/agg_internal.h cell/reader/finalizer, and the frozen Sort +
//  HashTable — so the ONLY difference from the real operator is one deliberately-
//  broken step, selected by the DISTINCTLY-NAMED `WindowMutation` enum (so it never
//  clashes with the other qe::mutant::*Mutation enums at link). The mutation
//  self-test (tests/window_mutation_test.cpp) runs the real operator and each mutant
//  through the SAME differential (independent reference + DuckDB) and shows the diff
//  CATCHES each mutant while the real operator passes.
//
//  Planted mutants (the §5 / §12 windowed-aggregation hazard taxonomy):
//   * kFrameOffByOne — the sliding frame includes P+1 preceding rows (lo = i-P-1
//     instead of i-P): the classic window-frame boundary bug, so every running
//     aggregate sees one extra row at the back of its frame.
//   * kBucketEdge    — the tumbling bucket edge is shifted by one
//     (bucket_id = (t + 1) / W): rows near a bucket boundary fall in the WRONG
//     bucket, so the grouping AND the emitted bucket_start diverge.
//
//  This is NOT linked into any engine target — only the WP-13 mutation self-test and
//  the consolidated catalog meta-test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/aggregate.h"  // AggSpec
#include "ops/hashtable.h"  // HashPath
#include "ops/join.h"       // GatherPath
#include "ops/operator.h"
#include "tsx/window.h"  // WindowMode

namespace qe::mutant {

enum class WindowMutation {
    kFrameOffByOne,
    kBucketEdge,
};

// Mirror of qe::tsx::Window with a planted defect.
class Window : public Operator {
   public:
    static constexpr std::size_t kOutBatch = 2048;

    Window(std::unique_ptr<Operator> child, qe::tsx::WindowMode mode,
           std::vector<std::uint32_t> keys, std::uint32_t time,
           std::int64_t param, std::vector<AggSpec> aggs, WindowMutation mut);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    void build_tumbling();
    void build_sliding();
    std::optional<Batch> next_tumbling();
    std::optional<Batch> next_sliding();

    std::unique_ptr<Operator> child_;
    qe::tsx::WindowMode mode_;
    std::vector<std::uint32_t> keys_;
    std::uint32_t time_;
    std::int64_t param_;
    std::vector<AggSpec> aggs_;
    WindowMutation mut_;
    Schema child_schema_;
    HashPath hash_path_ = HashPath::kVector;
    GatherPath gather_path_ = GatherPath::kVector;

    struct State;
    std::shared_ptr<State> state_;
    std::size_t emit_cursor_ = 0;
    OwnedBatch current_;
    bool opened_ = false;
};

}  // namespace qe::mutant
