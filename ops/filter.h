//  WP-3: Filter — evaluate a BOOL predicate over each child batch and emit the
//  passing rows. FIRST CUT COMPACTS (decision D6): it materializes a fresh dense
//  OwnedBatch of only the passing rows (no selection vector on the output).
//  Selection-vector flow is a later, measured optimization; compaction keeps the
//  first oracle simple while still being correct.
//
//  Semantics (must match DuckDB for the differential):
//   * The predicate is an expr tree of result type BOOL, evaluated through
//     expr::evaluate(), which APPLIES the input batch's selection vector and
//     returns a dense column of row_count values. So Filter handles an input
//     batch that already carries a selection vector for free.
//   * A row passes iff the predicate is TRUE. A NULL predicate result => the row
//     does NOT pass (SQL WHERE semantics).
//   * Compaction is out-of-place (via WP-1 compact_column), so the §12
//     selection-vector aliasing hazard cannot arise.
//   * Empty result batches are never emitted: Filter pulls the next child batch
//     until it has >=1 passing row or the child is exhausted (then nullopt).
#pragma once

#include <memory>
#include <optional>

#include "core/column.h"
#include "core/owned_batch.h"
#include "expr/expr.h"
#include "ops/operator.h"

namespace qe {

class Filter : public Operator {
   public:
    // `predicate` must have result type BOOL (checked at open()). Takes ownership
    // of the child operator.
    Filter(std::unique_ptr<Operator> child, expr::Expr predicate);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    std::unique_ptr<Operator> child_;
    expr::Expr predicate_;
    OwnedBatch current_;  // backs the view returned by the last next()
};

}  // namespace qe
