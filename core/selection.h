//  WP-1: selection-vector semantics (decision D6). Inline helpers that encode the
//  one rule every operator obeys: `sel == nullptr` means the dense identity
//  (logical row k == physical row k); otherwise logical row k is physical row
//  sel->idx[k], and the logical row count is sel->len.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/column.h"

namespace qe {

// Physical row index of logical row k. Precondition: k < selected_count(sel,*).
inline std::uint32_t sel_at(const SelectionVector* sel, std::size_t k) {
    return sel ? sel->idx[k] : static_cast<std::uint32_t>(k);
}

// Number of logical rows given a selection vector over columns of physical
// length `physical_len`. Dense (sel == nullptr) => physical_len.
inline std::size_t selected_count(const SelectionVector* sel,
                                  std::size_t physical_len) {
    return sel ? sel->len : physical_len;
}

}  // namespace qe
