//  WP-7: TEST-ONLY planted-mutant Sort operator. "A checker that cannot fail
//  proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::Sort is a faithful copy of the real Sort's drain / permutation /
//  emit loops (ops/sort.cpp) that REUSES the exact shared helpers
//  (ops/sort_internal.h: materialize / gather_rows / radix_eligible) — so the ONLY
//  difference from the real operator is one deliberately-broken step, selected by
//  the `Mutation` enum. The mutation self-test (tests/sort_mutation_test.cpp)
//  drives the real operator and each mutant through the SAME ORDER BY differential
//  under POSITIONAL compare (D12) and shows the diff CATCHES each mutant while the
//  real operator passes.
//
//  Planted mutants (each must be caught under POSITIONAL compare — a canonicalizing
//  comparator would hide a wrong row order, which is the whole point of the
//  ordered-mode comparator this WP adds):
//   * kDescSortsAsc      — the comparison path ignores DESC and sorts ASC.
//   * kNullsFlipped      — the comparison path flips NULLS FIRST/LAST.
//   * kRadixSignBug      — the radix encode omits the sign-bit flip, so negative
//     integers sort AFTER non-negatives (the classic radix high-byte/sign bug).
//   * kUnstableTiebreak  — equal-key rows are emitted in REVERSED original order
//     (breaks the stability the reference oracle relies on; observable on a
//     partial-key sort).
//   * kEmitTailOffByOne  — the output emit drops the LAST sorted row (short final
//     batch -> row-count mismatch).
//
//  NOT linked into any engine target — only the mutation self-test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/operator.h"
#include "ops/sort.h"  // SortKey

namespace qe::mutant {

enum class SortMutation {
    kDescSortsAsc,
    kNullsFlipped,
    kRadixSignBug,
    kUnstableTiebreak,
    kEmitTailOffByOne,
};

class Sort : public Operator {
   public:
    static constexpr std::size_t kOutBatch = 2048;

    Sort(std::unique_ptr<Operator> child, std::vector<SortKey> keys,
         SortMutation mut);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    std::unique_ptr<Operator> child_;
    std::vector<SortKey> keys_;
    SortMutation mut_;
    Schema child_schema_;

    struct State;
    std::shared_ptr<State> state_;
    std::size_t emit_cursor_ = 0;
    OwnedBatch current_;
    bool opened_ = false;
};

}  // namespace qe::mutant
