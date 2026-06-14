//  WP-3: Table implementation. See ops/table.h.
#include "ops/table.h"

#include <cassert>

namespace qe {

Table::Table(Schema schema, std::vector<OwnedColumn> cols)
    : schema_(std::move(schema)), cols_(std::move(cols)) {
    assert(cols_.size() == schema_.fields.size() &&
           "Table: column count must match schema field count");
    num_rows_ = cols_.empty() ? 0 : cols_[0].len();
    for (const auto& c : cols_) {
        assert(c.len() == num_rows_ && "Table: all columns must be equal length");
        (void)c;
    }
}

Batch Table::full_batch_view() const {
    Batch b;
    b.cols.reserve(cols_.size());
    for (const auto& c : cols_) b.cols.push_back(c.view());
    b.sel = nullptr;
    b.row_count = num_rows_;
    return b;
}

}  // namespace qe
