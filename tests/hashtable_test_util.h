//  WP-4 test helpers: build typed OwnedColumns and assemble a KeyColumns view
//  over them (optionally with a selection vector) to drive HashTable directly.
#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/hashtable.h"

namespace qe::ht_test {

// Build an OwnedColumn of `Type t` from `vals` (one int64 per value, reinterpreted
// to the column's physical type); indices in `nulls` are marked NULL. F64 columns
// take the value already bit_cast into the int64 slot — use f64_bits() below.
inline OwnedColumn make_col(Type t, const std::vector<std::int64_t>& bits,
                            const std::vector<std::size_t>& nulls = {}) {
    const std::size_t n = bits.size();
    OwnedColumn c = OwnedColumn::make(t, n);
    std::byte* d = c.mutable_data();
    for (std::size_t i = 0; i < n; ++i) {
        switch (t) {
            case Type::I32: {
                auto v = static_cast<std::int32_t>(bits[i]);
                std::memcpy(d + i * 4, &v, 4);
                break;
            }
            case Type::BOOL: {
                auto v = static_cast<std::uint8_t>(bits[i] ? 1 : 0);
                std::memcpy(d + i * 1, &v, 1);
                break;
            }
            case Type::I64:
            case Type::TS:
            case Type::F64: {
                std::int64_t v = bits[i];
                std::memcpy(d + i * 8, &v, 8);
                break;
            }
        }
    }
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

// Reinterpret a double as the int64 slot make_col(F64,...) expects.
inline std::int64_t f64_bits(double x) {
    std::int64_t v;
    std::memcpy(&v, &x, 8);
    return v;
}

// Holds the OwnedColumns and the Column-view array so a KeyColumns can point at
// stable storage for the call.
struct KeyHolder {
    std::vector<OwnedColumn> owned;
    std::vector<Column> views;
    std::vector<std::uint32_t> sel_idx;
    SelectionVector sel{nullptr, 0};
    bool has_sel = false;

    void set_selection(std::vector<std::uint32_t> idx) {
        sel_idx = std::move(idx);
        has_sel = true;
    }

    KeyColumns kc() {
        views.clear();
        views.reserve(owned.size());
        for (const auto& c : owned) views.push_back(c.view());
        if (has_sel) {
            sel = SelectionVector{sel_idx.data(), sel_idx.size()};
            return KeyColumns{views.data(), views.size(), &sel};
        }
        return KeyColumns{views.data(), views.size(), nullptr};
    }
};

}  // namespace qe::ht_test
