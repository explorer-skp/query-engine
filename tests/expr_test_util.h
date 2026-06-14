//  WP-2 test helpers: random batch generation (seeded), column equality that
//  respects the validity precedence, and a NaN-aware double compare. Shared by
//  the scalar==vector, null-truth-table, property, and mutation tests.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "core/validity.h"

namespace qe::expr::test {

// Bitwise-or-both-NaN double equality (scalar and vector paths are elementwise
// IEEE, so finite results are bit-identical; NaN bit patterns may differ).
inline bool double_eq(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

inline bool valid_at(const Column& c, std::size_t i) {
    return c.all_valid || c.validity == nullptr || qe::validity::get_bit(c.validity, i);
}

// Equal iff same type/len, identical validity at every row, and identical values
// at every VALID row (null-row payload is don't-care — that is the semantics).
inline bool col_equal(const OwnedColumn& x, const OwnedColumn& y) {
    if (x.type() != y.type() || x.len() != y.len()) return false;
    const Column cx = x.view(), cy = y.view();
    const std::size_t n = x.len();
    for (std::size_t i = 0; i < n; ++i) {
        if (valid_at(cx, i) != valid_at(cy, i)) return false;
        if (!valid_at(cx, i)) continue;
        switch (x.type()) {
            case Type::I32:
                if (reinterpret_cast<const std::int32_t*>(cx.data)[i] !=
                    reinterpret_cast<const std::int32_t*>(cy.data)[i])
                    return false;
                break;
            case Type::I64:
            case Type::TS:
                if (reinterpret_cast<const std::int64_t*>(cx.data)[i] !=
                    reinterpret_cast<const std::int64_t*>(cy.data)[i])
                    return false;
                break;
            case Type::F64:
                if (!double_eq(reinterpret_cast<const double*>(cx.data)[i],
                               reinterpret_cast<const double*>(cy.data)[i]))
                    return false;
                break;
            case Type::BOOL:
                if (reinterpret_cast<const std::uint8_t*>(cx.data)[i] !=
                    reinterpret_cast<const std::uint8_t*>(cy.data)[i])
                    return false;
                break;
        }
    }
    return true;
}

// A random OwnedColumn of `n` values of `type`, ~`null_pct`% nulls. Integer data
// spans the full range (exercises wraparound); F64 is moderate-magnitude with
// occasional ±inf/NaN; BOOL is 0/1.
inline OwnedColumn random_column(std::mt19937_64& rng, Type type, std::size_t n,
                                 int null_pct = 20) {
    OwnedColumn c = OwnedColumn::make(type, n);
    void* d = c.mutable_data();
    for (std::size_t i = 0; i < n; ++i) {
        switch (type) {
            case Type::I32:
                static_cast<std::int32_t*>(d)[i] =
                    static_cast<std::int32_t>(rng());
                break;
            case Type::I64:
            case Type::TS:
                static_cast<std::int64_t*>(d)[i] =
                    static_cast<std::int64_t>(rng());
                break;
            case Type::F64: {
                const std::uint64_t r = rng() % 100;
                double v;
                if (r == 0)
                    v = std::numeric_limits<double>::infinity();
                else if (r == 1)
                    v = -std::numeric_limits<double>::infinity();
                else if (r == 2)
                    v = std::nan("");
                else
                    v = (static_cast<double>(static_cast<std::int64_t>(rng())) /
                         static_cast<double>(1ull << 40));
                static_cast<double*>(d)[i] = v;
                break;
            }
            case Type::BOOL:
                static_cast<std::uint8_t*>(d)[i] = (rng() & 1u) ? 1 : 0;
                break;
        }
        if (null_pct > 0 && static_cast<int>(rng() % 100) < null_pct)
            c.set_null(i);
    }
    return c;
}

// Wrap a set of OwnedColumns into an OwnedBatch (optionally with a selection
// vector). The OwnedBatch owns the columns; view() yields the frozen Batch.
inline OwnedBatch make_batch(std::vector<OwnedColumn> cols) {
    OwnedBatch b;
    for (auto& c : cols) b.add_column(std::move(c));
    return b;
}

}  // namespace qe::expr::test
