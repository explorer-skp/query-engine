//  WP-3: Scan — the leaf operator. Yields a Table (ops/table.h) as a sequence of
//  dense <=batch_size-row Batches whose Column views point ZERO-COPY into the
//  Table's storage. No filtering, no projection: just hand the columns out in
//  batch-sized windows.
//
//  WHY batch_size must be a multiple of 64 (asserted): the validity bitmap is
//  bit-packed, 64 values per word (core/validity.h). A batch starting at row
//  `start` exposes its validity as a pointer offset of start/64 words into the
//  column's bitmap — which is only a valid (bit-0-aligned) sub-view when `start`
//  is a multiple of 64. With batch_size % 64 == 0, every batch start
//  (k*batch_size) is a multiple of 64, so the offset is always exact and the
//  scan stays zero-copy. (64 is the validity WORD width from the format, not an
//  ISA/vector width — no Mac->x86 hazard.) The TAIL batch may have a row count
//  that is NOT a multiple of 64; that only affects its own `len`, and bits past
//  `len` in the final word are padding the contract forbids consumers to read.
#pragma once

#include <cstddef>
#include <optional>

#include "core/column.h"
#include "ops/operator.h"
#include "ops/table.h"

namespace qe {

class Scan : public Operator {
   public:
    // Default batch size is decision D4 (2048), configurable for the batch-size
    // sweep. Must be a positive multiple of 64 (see header comment).
    static constexpr std::size_t kDefaultBatchSize = 2048;

    explicit Scan(const Table& table,
                  std::size_t batch_size = kDefaultBatchSize);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    const Table& table_;
    std::size_t batch_size_;
    std::size_t cursor_ = 0;  // next physical row to emit
};

}  // namespace qe
