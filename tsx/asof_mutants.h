//  WP-12: TEST-ONLY planted-mutant as-of-join operator. "A checker that cannot fail
//  proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::AsofJoin is a faithful copy of the real as-of operator's build /
//  merge / output loops (tsx/asof.cpp) that REUSES the exact shared gather emitters
//  (ops/join_internal.h: BuildStore + emit_probe_column/emit_build_column) and the
//  frozen Sort + HashTable — so the ONLY difference from the real operator is one
//  deliberately-broken step, selected by the `Mutation` enum. The mutation
//  self-test (tests/asof_mutation_test.cpp) runs the real operator and a mutant
//  through the SAME differential (independent reference + DuckDB) and shows the diff
//  CATCHES each mutant while the real operator passes.
//
//  Planted mutants (the §5 / §12 as-of hazard taxonomy):
//   * kBoundaryStrict     — advance with `<` instead of `<=`, so a build row whose
//     timestamp EQUALS the probe timestamp is DROPPED (the classic `>`-vs-`>=`
//     boundary bug — equal-timestamp matches lost).
//   * kNearestFollowing   — emit the nearest FOLLOWING build row (cursor) instead of
//     the nearest preceding (cursor-1): forward as-of instead of backward.
//   * kIgnoreLastKey      — build/probe the partition table over all keys EXCEPT the
//     last, so a composite-key as-of over-matches across the dropped key.
//   * kLeftWrongNull      — a LEFT-join unmatched probe row emits build row 0's real
//     (valid) values instead of NULLs.
//
//  This is NOT linked into any engine target — only the as-of mutation self-test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/hashtable.h"  // HashPath
#include "ops/join.h"       // GatherPath
#include "ops/operator.h"
#include "tsx/asof.h"  // AsofType

namespace qe::mutant {

enum class AsofMutation {
    kBoundaryStrict,
    kNearestFollowing,
    kIgnoreLastKey,
    kLeftWrongNull,
};

// Mirror of qe::tsx::AsofJoin with a planted defect.
class AsofJoin : public Operator {
   public:
    static constexpr std::size_t kOutBatch = 2048;

    AsofJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
             std::vector<std::uint32_t> probe_keys,
             std::vector<std::uint32_t> build_keys, std::uint32_t probe_time,
             std::uint32_t build_time, qe::tsx::AsofType type,
             std::optional<std::int64_t> tolerance, AsofMutation mut);

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
    std::uint32_t probe_time_;
    std::uint32_t build_time_;
    qe::tsx::AsofType type_;
    std::optional<std::int64_t> tolerance_;
    AsofMutation mut_;
    Schema probe_schema_;
    Schema build_schema_;
    HashPath hash_path_ = HashPath::kVector;
    GatherPath gather_path_ = GatherPath::kVector;

    struct State;
    std::shared_ptr<State> state_;
    std::optional<Batch> cur_probe_;
    std::size_t pair_cursor_ = 0;
    OwnedBatch current_;
    bool opened_ = false;
    bool probe_done_ = false;
};

}  // namespace qe::mutant
