//  WP-1: the OWNING layer that allocates Buffers and hands out the frozen,
//  non-owning Column/Batch views (core/column.h). Operators build results here
//  and return view() objects; the OwnedColumn/OwnedBatch keep the bytes alive.
//
//  These are NEW types the format needs (the frozen view structs cannot own
//  storage — they hold raw pointers). They enforce the validity precedence from
//  core/column.h: a column with no nulls reports all_valid == true and a null
//  validity pointer; a column with a null carries an allocated bitmap and
//  all_valid == false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/buffer.h"
#include "core/column.h"
#include "core/string_dict.h"
#include "core/types.h"

namespace qe {

// One owned column: a data Buffer, an optional validity Buffer, and the
// all_valid flag. Construct via make()/make_nullable(), fill through the mutable
// accessors, then publish a view().
class OwnedColumn {
   public:
    OwnedColumn() = default;

    // Non-nullable column: `len` values, data uninitialized, no validity buffer,
    // all_valid == true. Marking a null later (set_null) lazily allocates the
    // bitmap.
    static OwnedColumn make(Type type, std::size_t len);

    // WP-7b: a Type::STR column of `len` int32 codes, all-valid, carrying `dict`
    // (the code->bytes table). Fill the codes via mutable_data() (as int32) and
    // intern through the dict; the codes must be valid indices into `dict`.
    static OwnedColumn make_str(std::size_t len,
                                std::shared_ptr<const StringDict> dict);

    Type type() const noexcept { return type_; }
    std::size_t len() const noexcept { return len_; }
    bool all_valid() const noexcept { return all_valid_; }

    std::byte* mutable_data() noexcept { return data_.data(); }
    const std::byte* data() const noexcept { return data_.data(); }

    // The validity bitmap words, or nullptr if none has been allocated. Allocate
    // one (all bits valid) with ensure_validity().
    std::uint64_t* mutable_validity() noexcept {
        return validity_.empty()
                   ? nullptr
                   : reinterpret_cast<std::uint64_t*>(validity_.data());
    }
    const std::uint64_t* validity() const noexcept {
        return validity_.empty()
                   ? nullptr
                   : reinterpret_cast<const std::uint64_t*>(validity_.data());
    }

    // Allocate the validity bitmap if absent, initialized all-valid. Idempotent.
    void ensure_validity();

    // Mark value i null: allocates the bitmap if needed, clears bit i, and sets
    // all_valid = false.
    void set_null(std::size_t i);

    // Recompute all_valid from the current bitmap (no-op if no bitmap). Use after
    // bulk-editing the validity words directly.
    void refresh_all_valid();

    // WP-7b STR dictionary. A STR OwnedColumn MUST carry a dict before view() is
    // called (view() publishes it as Column::dict). Two ways to attach one:
    //   * set_dict(shared)     — this column shares OWNERSHIP of the dict (source
    //     columns and operator-derived dicts, e.g. aggregate group keys).
    //   * set_dict_ref(raw)    — this column merely REFERENCES a dict owned and
    //     kept alive elsewhere (gathered/compacted subsets reuse their source's
    //     dict; the codes are unchanged so they stay valid in that same dict).
    // dict() exposes the raw pointer view() will publish (nullptr if unset).
    void set_dict(std::shared_ptr<const StringDict> d) { dict_ = std::move(d); }
    void set_dict_ref(const StringDict* d) {
        // Aliasing shared_ptr: shares the pointer, owns nothing (caller guarantees
        // lifetime, exactly as for data()/validity() backing storage).
        dict_ = std::shared_ptr<const StringDict>(std::shared_ptr<void>{}, d);
    }
    const StringDict* dict() const noexcept { return dict_.get(); }

    // Frozen view applying the precedence rule: validity is exposed only when
    // all_valid is false.
    Column view() const;

   private:
    Type type_ = Type::I32;
    std::size_t len_ = 0;
    Buffer data_;
    Buffer validity_;  // empty() => no bitmap allocated
    bool all_valid_ = true;
    // WP-7b: non-null only for Type::STR (owning or aliasing per set_dict*).
    std::shared_ptr<const StringDict> dict_;
};

// One owned batch: a set of OwnedColumns plus optional owned selection-vector
// storage. view() produces a frozen Batch whose Column views and SelectionVector
// point into this object — valid for the OwnedBatch's lifetime.
class OwnedBatch {
   public:
    void add_column(OwnedColumn col) { cols_.push_back(std::move(col)); }
    OwnedColumn& column(std::size_t i) { return cols_[i]; }
    const OwnedColumn& column(std::size_t i) const { return cols_[i]; }
    std::size_t num_columns() const noexcept { return cols_.size(); }

    // Install an owned selection vector (copied in). Subsequent view()s are
    // selected; row_count becomes idx.size().
    void set_selection(std::vector<std::uint32_t> idx);
    void clear_selection() noexcept;

    Batch view() const;

   private:
    std::vector<OwnedColumn> cols_;
    std::vector<std::uint32_t> sel_idx_;
    bool has_sel_ = false;
    // Stable backing for the SelectionVector a view() hands out; refreshed each
    // view() so it always reflects sel_idx_. Address is stable for this object's
    // lifetime. mutable: view() is logically const.
    mutable SelectionVector sel_view_{nullptr, 0};
};

// Compact `in` under selection vector `sel` (nullptr => dense identity copy) into
// a fresh, dense OwnedColumn of `n` values, where out value k = in value
// sel_at(sel, k). OUT-OF-PLACE by construction (the result owns new buffers), so
// the §12 selection-vector aliasing hazard cannot arise. Preserves nullness:
// the result is all_valid iff every selected input value was valid.
OwnedColumn compact_column(const Column& in, const SelectionVector* sel,
                           std::size_t n);

}  // namespace qe
