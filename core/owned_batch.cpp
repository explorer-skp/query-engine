//  WP-1: OwnedColumn / OwnedBatch / compact_column. See core/owned_batch.h.

#include "core/owned_batch.h"

#include <cassert>

#include "core/selection.h"
#include "core/validity.h"
#include "simd/gather_kernels.h"
#include "simd/validity_kernels.h"

namespace qe {

// ----- OwnedColumn -----

OwnedColumn OwnedColumn::make(Type type, std::size_t len) {
    OwnedColumn c;
    c.type_ = type;
    c.len_ = len;
    c.data_ = Buffer(len * byte_width(type));
    c.all_valid_ = true;
    return c;
}

void OwnedColumn::ensure_validity() {
    if (!validity_.empty() || len_ == 0) return;
    validity_ = Buffer(validity::words(len_) * sizeof(std::uint64_t));
    validity::fill_all_valid(reinterpret_cast<std::uint64_t*>(validity_.data()),
                             len_);
}

void OwnedColumn::set_null(std::size_t i) {
    assert(i < len_ && "set_null index out of range");
    ensure_validity();
    validity::set_bit(reinterpret_cast<std::uint64_t*>(validity_.data()), i,
                      false);
    all_valid_ = false;
}

void OwnedColumn::refresh_all_valid() {
    if (validity_.empty()) {
        all_valid_ = true;
        return;
    }
    all_valid_ = simd::all_valid_vec(
        reinterpret_cast<const std::uint64_t*>(validity_.data()), len_);
}

Column OwnedColumn::view() const {
    Column c;
    c.type = type_;
    c.len = len_;
    c.data = data_.data();
    // Precedence (core/column.h): expose the bitmap only when there are nulls.
    c.all_valid = all_valid_;
    c.validity = all_valid_ ? nullptr : validity();
    return c;
}

// ----- OwnedBatch -----

void OwnedBatch::set_selection(std::vector<std::uint32_t> idx) {
    sel_idx_ = std::move(idx);
    has_sel_ = true;
}

void OwnedBatch::clear_selection() noexcept {
    sel_idx_.clear();
    has_sel_ = false;
}

Batch OwnedBatch::view() const {
    Batch b;
    b.cols.reserve(cols_.size());
    for (const auto& c : cols_) b.cols.push_back(c.view());
    if (has_sel_) {
        // The SelectionVector must outlive the returned Batch view; sel_view_ is
        // owned by this OwnedBatch and its address is stable for our lifetime.
        sel_view_.idx = sel_idx_.data();
        sel_view_.len = sel_idx_.size();
        b.sel = &sel_view_;
        b.row_count = sel_idx_.size();
    } else {
        b.sel = nullptr;
        b.row_count = cols_.empty() ? 0 : cols_.front().len();
    }
    return b;
}

// ----- compaction -----

OwnedColumn compact_column(const Column& in, const SelectionVector* sel,
                           std::size_t n) {
    OwnedColumn out = OwnedColumn::make(in.type, n);
    const std::uint32_t* idx = sel ? sel->idx : nullptr;

    // Gather the data by lane width. out and in own distinct Buffers => no alias.
    assert(out.data() != in.data && "compaction must be out-of-place");
    switch (byte_width(in.type)) {
        case 1:
            simd::gather8_scalar(
                reinterpret_cast<const std::uint8_t*>(in.data), idx, n,
                reinterpret_cast<std::uint8_t*>(out.mutable_data()));
            break;
        case 4:
            simd::gather32_vec(
                reinterpret_cast<const std::uint32_t*>(in.data), idx, n,
                reinterpret_cast<std::uint32_t*>(out.mutable_data()));
            break;
        case 8:
            simd::gather64_vec(
                reinterpret_cast<const std::uint64_t*>(in.data), idx, n,
                reinterpret_cast<std::uint64_t*>(out.mutable_data()));
            break;
        default:
            assert(false && "unsupported value width");
    }

    // Gather validity (scalar bit-gather). all_valid input => all_valid output.
    if (!in.all_valid && in.validity != nullptr) {
        out.ensure_validity();
        std::uint64_t* ov =
            reinterpret_cast<std::uint64_t*>(out.mutable_validity());
        for (std::size_t k = 0; k < n; ++k) {
            const bool v = validity::get_bit(in.validity, sel_at(sel, k));
            validity::set_bit(ov, k, v);
        }
        out.refresh_all_valid();
    }
    return out;
}

}  // namespace qe
