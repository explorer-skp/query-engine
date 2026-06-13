//  DRAFT — NOT FROZEN. Owned by the final review. Frozen at WP-1 (core/simd) /
//  WP-3 (Operator). Do not add fields, rename, or implement logic here.
//
//  WP-0 scaffold: non-binding signature stub. The typed column batch that flows
//  through every pull-based vectorized operator (~2048-value batches). No logic
//  lives here yet — only the shape of the data contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"

namespace qe {

struct Column {
    Type type;
    size_t len;
    const std::byte* data;     // contiguous, type-width
    const uint64_t* validity;  // nullptr => no nulls present
    bool all_valid;            // fast-path flag
};

struct SelectionVector {
    const uint32_t* idx;
    size_t len;
};

struct Batch {
    std::vector<Column> cols;
    const SelectionVector* sel;  // nullptr => dense 0..len-1
    size_t row_count;            // logical rows (respects sel)
};

struct Schema {
    std::vector<std::pair<std::string, Type>> fields;
};

}  // namespace qe
