//  WP-14: TEST-ONLY planted-mutant compressed scan. "A checker that cannot fail
//  proves nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::CompressedScan is a faithful copy of the real streaming decoder
//  (tsx/compress.cpp) over the SAME EncodedColumn / CompressedTable format and the
//  SAME zig-zag kernel — differing by exactly ONE deliberately-broken decode step,
//  selected by the DISTINCTLY-named `CompressMutation` enum (so it never clashes
//  with the other qe::mutant::*Mutation enums at link). The round-trip / compressed-
//  scan differential is shown to CATCH each mutant while the real decoder passes.
//
//  Planted decode bit-slips (≥2, per the WP brief):
//   * kDropSecondDerivative   — the delta-of-delta reconstruction omits the second-
//     difference term (treats each dod as a first-order delta), so reconstructed
//     TS/I64 timestamps DRIFT after the second value.
//   * kGorillaLeadingZerosOff — the Gorilla leading-zero count is taken off by one
//     when re-deriving the trailing-zero shift, so a reconstructed double is wrong
//     by bits (whenever the encoded block had ≥1 leading zero).
//
//  NOT linked into any engine target — only the WP-14 mutation self-test and the
//  consolidated catalog meta-test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "core/column.h"
#include "core/owned_batch.h"
#include "ops/operator.h"
#include "ops/scan.h"       // Scan::kDefaultBatchSize
#include "tsx/compress.h"   // CompressedTable / EncodedColumn

namespace qe::mutant {

enum class CompressMutation {
    kDropSecondDerivative,
    kGorillaLeadingZerosOff,
};

// Mirror of qe::tsx::CompressedScan with a planted decode defect.
class CompressedScan : public Operator {
   public:
    CompressedScan(const qe::tsx::CompressedTable& ct, CompressMutation mut,
                  std::size_t batch_size = Scan::kDefaultBatchSize);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    const qe::tsx::CompressedTable& ct_;
    CompressMutation mut_;
    std::size_t batch_size_;
    std::size_t cursor_ = 0;

    struct State;
    std::shared_ptr<State> state_;
    OwnedBatch current_;
    bool opened_ = false;
};

}  // namespace qe::mutant
