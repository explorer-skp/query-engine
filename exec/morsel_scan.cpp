//  WP-10b: MorselScan implementation. See exec/morsel_scan.h. The per-batch view
//  construction is intentionally identical to ops/scan.cpp (zero-copy column views,
//  whole-word validity offset) — only the row window [row_start_, row_end_) and the
//  64-aligned start precondition differ.
#include "exec/morsel_scan.h"

#include <algorithm>
#include <stdexcept>

#include "core/types.h"

namespace qe::exec {

MorselScan::MorselScan(const Table& table, std::size_t row_start,
                       std::size_t row_end, std::size_t batch_size)
    : table_(table),
      row_start_(row_start),
      row_end_(row_end),
      batch_size_(batch_size) {
    // Real preconditions (not asserts) so the zero-copy validity-word-alignment
    // invariant always holds, in every build.
    if (batch_size_ == 0 || batch_size_ % 64 != 0)
        throw std::invalid_argument(
            "MorselScan batch_size must be a positive multiple of 64");
    if (row_start_ % 64 != 0)
        throw std::invalid_argument(
            "MorselScan row_start must be a multiple of 64 (validity word "
            "alignment — see morsel_scan.h)");
    if (row_end_ > table_.num_rows() || row_start_ > row_end_)
        throw std::invalid_argument("MorselScan row range out of bounds");
}

void MorselScan::open() { cursor_ = row_start_; }

void MorselScan::close() { cursor_ = row_end_; }

Schema MorselScan::output_schema() const { return table_.schema(); }

std::optional<Batch> MorselScan::next() {
    if (cursor_ >= row_end_) return std::nullopt;

    const std::size_t start = cursor_;
    const std::size_t n = std::min(batch_size_, row_end_ - start);

    Batch b;
    b.cols.reserve(table_.num_columns());
    for (std::size_t c = 0; c < table_.num_columns(); ++c) {
        const OwnedColumn& oc = table_.column(c);
        Column view{};
        view.type = oc.type();
        view.len = n;
        view.data = oc.data() + start * byte_width(oc.type());
        // WP-7b: carry the STR dictionary into the batch view (nullptr for
        // non-STR), exactly as ops/scan.cpp does. Omitting this line left
        // view.dict indeterminate for STR columns — every downstream STR
        // consumer then dereferenced a wild pointer (audit C1a).
        view.dict = oc.dict();
        if (oc.all_valid()) {
            view.validity = nullptr;
            view.all_valid = true;
        } else {
            // start is a multiple of 64, so a whole-word offset re-bases the
            // bitmap exactly (view bit i == column bit start+i).
            view.validity = oc.validity() + (start / 64);
            view.all_valid = false;
        }
        b.cols.push_back(view);
    }
    b.sel = nullptr;  // morsel output is dense
    b.row_count = n;

    cursor_ = start + n;
    return b;
}

}  // namespace qe::exec
