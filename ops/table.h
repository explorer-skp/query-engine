//  WP-3: the in-memory columnar source the Scan operator reads from. This is the
//  "table/source shape" — owned by this WP (WP-8's plan layer will later drive
//  its construction). Deliberately minimal: a Schema (named, typed columns) plus
//  one OwnedColumn per field, all of equal length. It is built on the WP-1 owning
//  layer (core/owned_batch.h) so every byte it hands out as a Column view stays
//  alive for the Table's lifetime.
//
//  Ownership: a Table OWNS its column storage (OwnedColumn). Scan yields Batches
//  whose Column views point into this storage, so a Table must outlive any
//  operator tree scanning it.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"

namespace qe {

class Table {
   public:
    // Build a table from a schema and matching owned columns. Every column's
    // len() must equal `num_rows` (checked in debug); cols.size() must equal the
    // schema's field count. Takes ownership of the columns.
    Table(Schema schema, std::vector<OwnedColumn> cols);

    const Schema& schema() const noexcept { return schema_; }
    std::size_t num_rows() const noexcept { return num_rows_; }
    std::size_t num_columns() const noexcept { return cols_.size(); }

    const OwnedColumn& column(std::size_t i) const { return cols_[i]; }

    // A dense, whole-table Batch view (no selection vector). Used by the
    // reference oracle, which computes results over the whole table in one shot.
    // The returned Batch's Column views are valid for this Table's lifetime; the
    // returned std::vector<Column> backing is owned by the Batch itself.
    Batch full_batch_view() const;

   private:
    Schema schema_;
    std::vector<OwnedColumn> cols_;
    std::size_t num_rows_ = 0;
};

}  // namespace qe
