//  WP-10b: MorselScan — the additive parallel layer's leaf operator. It is a
//  Scan (ops/scan.h) RESTRICTED to a contiguous row range [row_start, row_end) of
//  a Table — a "morsel" (a scan partition; see the glossary). It yields the same
//  zero-copy, dense, <=batch_size-row Batches the frozen Scan does, but only over
//  its morsel, so independent worker threads can each drive a private operator
//  pipeline over a disjoint slice of the same Table.
//
//  This adds NO frozen-interface change: MorselScan implements the frozen
//  ops/operator.h pull contract and composes with the frozen operators unchanged.
//  It lives in the qe::exec namespace (the self-contained parallel layer), never in
//  ops/.
//
//  ALIGNMENT (same invariant as Scan, and why morsels are 64-aligned): the validity
//  bitmap is 64 values per word, so a batch's validity sub-view is only bit-0
//  aligned when its physical start is a multiple of 64. `row_start` MUST be a
//  multiple of 64 (enforced) and batch_size a positive multiple of 64, so every
//  emitted sub-batch starts at a multiple of 64. `row_end` may be the table tail
//  (not a multiple of 64); that only shortens the final sub-batch's len, and bits
//  past len are padding the contract forbids consumers to read — exactly the frozen
//  Scan's tail rule.
#pragma once

#include <cstddef>
#include <optional>

#include "core/column.h"
#include "ops/operator.h"
#include "ops/table.h"

namespace qe::exec {

class MorselScan : public Operator {
   public:
    // Scan rows [row_start, row_end) of `table`. row_start must be a multiple of
    // 64 and row_end <= table.num_rows(); batch_size a positive multiple of 64.
    MorselScan(const Table& table, std::size_t row_start, std::size_t row_end,
               std::size_t batch_size = 2048);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    const Table& table_;
    std::size_t row_start_;
    std::size_t row_end_;
    std::size_t batch_size_;
    std::size_t cursor_ = 0;  // next physical row to emit
};

}  // namespace qe::exec
