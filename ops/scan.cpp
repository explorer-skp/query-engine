//  WP-3: Scan implementation. See ops/scan.h.
#include "ops/scan.h"

#include <algorithm>
#include <stdexcept>

#include "core/types.h"

namespace qe {

Scan::Scan(const Table& table, std::size_t batch_size)
    : table_(table), batch_size_(batch_size) {
    // Real precondition check (not an assert): enforced in every build so the
    // word-aligned zero-copy validity sub-view invariant always holds.
    if (batch_size_ == 0 || batch_size_ % 64 != 0)
        throw std::invalid_argument(
            "Scan batch_size must be a positive multiple of 64 (validity word "
            "alignment — see scan.h)");
}

void Scan::open() { cursor_ = 0; }

void Scan::close() { cursor_ = table_.num_rows(); }

Schema Scan::output_schema() const { return table_.schema(); }

std::optional<Batch> Scan::next() {
    const std::size_t total = table_.num_rows();
    if (cursor_ >= total) return std::nullopt;

    const std::size_t start = cursor_;
    const std::size_t n = std::min(batch_size_, total - start);

    Batch b;
    b.cols.reserve(table_.num_columns());
    for (std::size_t c = 0; c < table_.num_columns(); ++c) {
        const OwnedColumn& oc = table_.column(c);
        Column view;
        view.type = oc.type();
        view.len = n;
        view.data = oc.data() + start * byte_width(oc.type());
        if (oc.all_valid()) {
            // Fast path: no nulls in the whole column => none in this window.
            view.validity = nullptr;
            view.all_valid = true;
        } else {
            // start is a multiple of 64 (batch_size_ is), so a whole-word offset
            // re-bases the bitmap so that view bit i == column bit (start+i).
            view.validity = oc.validity() + (start / 64);
            view.all_valid = false;
        }
        b.cols.push_back(view);
    }
    b.sel = nullptr;  // scan output is dense
    b.row_count = n;

    cursor_ = start + n;
    return b;
}

}  // namespace qe
