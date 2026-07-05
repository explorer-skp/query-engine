//  WP-5: INTERNAL aggregation helpers shared by the real Aggregate operator
//  (ops/aggregate.cpp) AND the test-only planted-mutant operator
//  (ops/aggregate_mutants.cpp). NOT a frozen contract (the analog of
//  ops/hash_internal.h). Centralizing the NON-mutated pieces here — the per-group
//  state cell, its initialization, the typed column reader, and the key/agg
//  output-column writers — lets the mutant be a faithful "real accumulate minus
//  one step": it reuses everything here UNCHANGED and copies only the
//  accumulate/finalize loop with a single planted defect, so the differential
//  isolates that defect.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/selection.h"
#include "core/types.h"
#include "core/validity.h"
#include "ops/aggregate.h"  // AggFunc, AggSpec, agg_result_type

namespace qe::ops::detail {

// F64 TOTAL-ORDER POLICY for MIN/MAX (matches DuckDB): NaN is GREATER than every
// other double (all NaNs tie). So MIN returns NaN only when every non-NULL input
// is NaN, and MAX returns NaN when any non-NULL input is NaN. This replaces raw
// std::min/std::max, whose NaN behavior silently drops NaNs, is position-
// dependent, and (in the Highway kernels) differs between NEON and AVX — the
// exact ISA-divergence audit finding C2. Under this order the MIN identity is
// NaN (the greatest element) and the MAX identity stays -inf.
inline bool f64_less_total(double a, double b) {
    if (std::isnan(b)) return !std::isnan(a);  // anything real < NaN; NaN !< NaN
    if (std::isnan(a)) return false;           // NaN !< real
    return a < b;
}
inline double f64_min_total(double a, double b) { return f64_less_total(b, a) ? b : a; }
inline double f64_max_total(double a, double b) { return f64_less_total(a, b) ? b : a; }

// Per-group, per-aggregate accumulator. `cnt` is the COUNT(*) row count for a
// CountStar agg, otherwise the NON-NULL input count (drives COUNT(col), the
// SUM/MIN/MAX "seen" test, and the AVG denominator). The integer accumulator `i`
// holds SUM/MIN/MAX/AVG-numerator for integer-family inputs; the double `d` holds
// them for F64 inputs. Exactly one of i/d is used per agg, picked by is_float().
struct AggCell {
    std::int64_t i = 0;
    double d = 0.0;
    std::int64_t cnt = 0;
};

// True iff agg `s`'s accumulator lives in `d` (F64 input column), false => `i`.
// CountStar/Count never touch i/d, so the answer is irrelevant for them.
inline bool agg_is_float(const AggSpec& s, Type input_type) {
    (void)s;
    return input_type == Type::F64;
}

// A fresh accumulator for a brand-new group: zero for COUNT/SUM/AVG, the
// op-identity (so a real value always wins) for MIN/MAX. Under the NaN-greatest
// total order the F64 MIN identity is NaN, not +inf: an all-NaN group must emit
// NaN (as DuckDB does), and every real value still beats the identity because
// f64_min_total picks it over NaN.
inline AggCell init_agg_cell(AggFunc func, bool is_float) {
    AggCell c;  // i=0, d=0, cnt=0
    switch (func) {
        case AggFunc::Min:
            c.i = std::numeric_limits<std::int64_t>::max();
            c.d = std::numeric_limits<double>::quiet_NaN();
            break;
        case AggFunc::Max:
            c.i = std::numeric_limits<std::int64_t>::min();
            c.d = -std::numeric_limits<double>::infinity();
            break;
        default:
            break;  // Count/CountStar/Sum/Avg start at zero
    }
    (void)is_float;
    return c;
}

// One typed value read from logical row k of a batch column under selection.
struct ColVal {
    bool valid = false;
    std::int64_t i = 0;
    double d = 0.0;
};

inline ColVal read_col(const Column& c, const SelectionVector* sel,
                       std::size_t k) {
    const std::uint32_t p = sel_at(sel, k);
    ColVal r;
    r.valid = c.all_valid || validity::get_bit(c.validity, p);
    if (!r.valid) return r;
    switch (c.type) {
        case Type::I32:
            r.i = reinterpret_cast<const std::int32_t*>(c.data)[p];
            break;
        case Type::I64:
        case Type::TS:
            r.i = reinterpret_cast<const std::int64_t*>(c.data)[p];
            break;
        case Type::BOOL:
            r.i = reinterpret_cast<const std::uint8_t*>(c.data)[p];
            break;
        case Type::F64:
            r.d = reinterpret_cast<const double*>(c.data)[p];
            break;
        case Type::STR:
            // WP-7b: the int32 dictionary code lands in `i`. COUNT(str) only needs
            // `valid`; MIN/MAX(str) compare by VALUE and are handled specially in
            // aggregate.cpp (this code alone is not order-meaningful — see report).
            r.i = reinterpret_cast<const std::int32_t*>(c.data)[p];
            break;
    }
    return r;
}

// Write the decoded key word of group `g`, column `col` (of Type `t`), read back
// from the hash table, into output OwnedColumn `out` at row `r`. `is_null`/`word`
// come from HashTable::group_is_null / group_key_word.
inline void write_key_cell(OwnedColumn& out, std::size_t r, Type t, bool is_null,
                           std::uint64_t word) {
    if (is_null) {
        out.set_null(r);
        return;
    }
    std::byte* d = out.mutable_data();
    switch (t) {
        case Type::I32: {
            auto v = static_cast<std::int32_t>(word);
            reinterpret_cast<std::int32_t*>(d)[r] = v;
            break;
        }
        case Type::I64:
        case Type::TS: {
            std::int64_t v;
            std::memcpy(&v, &word, 8);
            reinterpret_cast<std::int64_t*>(d)[r] = v;
            break;
        }
        case Type::F64: {
            double v;
            std::memcpy(&v, &word, 8);
            reinterpret_cast<double*>(d)[r] = v;
            break;
        }
        case Type::BOOL:
            reinterpret_cast<std::uint8_t*>(d)[r] =
                static_cast<std::uint8_t>(word & 1ull);
            break;
        case Type::STR:
            // WP-7b: the group-key word is a dictionary code (canonicalized into the
            // aggregate's owned dict); write it as the int32 code. The caller
            // attaches that owned dict to the output column.
            reinterpret_cast<std::int32_t*>(d)[r] =
                static_cast<std::int32_t>(word);
            break;
    }
}

// Write the FINALIZED aggregate value of cell `c` (agg `func`, F64 input?
// `is_float`, result Type `rt`) into output OwnedColumn `out` at row `r`. This is
// the canonical (correct) finalization; the mutant copies it with one defect.
//
// NULL/empty semantics: SUM/MIN/MAX/AVG over a group with zero non-null inputs
// (cnt==0) => NULL. COUNT(*)/COUNT(col) are never NULL.
inline void write_agg_cell(OwnedColumn& out, std::size_t r, AggFunc func,
                           bool is_float, Type rt, const AggCell& c) {
    std::byte* d = out.mutable_data();
    switch (func) {
        case AggFunc::CountStar:
        case AggFunc::Count:
            reinterpret_cast<std::int64_t*>(d)[r] = c.cnt;  // I64, never null
            return;
        case AggFunc::Sum:
            if (c.cnt == 0) {
                out.set_null(r);
                return;
            }
            if (rt == Type::F64)
                reinterpret_cast<double*>(d)[r] = c.d;
            else
                reinterpret_cast<std::int64_t*>(d)[r] = c.i;  // SUM(int)->I64
            return;
        case AggFunc::Min:
        case AggFunc::Max:
            if (c.cnt == 0) {
                out.set_null(r);
                return;
            }
            // result type == input type; write at that width from i/d.
            switch (rt) {
                case Type::I32:
                    reinterpret_cast<std::int32_t*>(d)[r] =
                        static_cast<std::int32_t>(c.i);
                    break;
                case Type::I64:
                case Type::TS:
                    reinterpret_cast<std::int64_t*>(d)[r] = c.i;
                    break;
                case Type::BOOL:
                    reinterpret_cast<std::uint8_t*>(d)[r] =
                        static_cast<std::uint8_t>(c.i & 1);
                    break;
                case Type::F64:
                    reinterpret_cast<double*>(d)[r] = c.d;
                    break;
                case Type::STR:
                    // WP-7b: MIN/MAX(str) result is a dictionary code (the canonical
                    // min/max string's code in the aggregate's owned dict, chosen by
                    // VALUE in aggregate.cpp); write it as int32.
                    reinterpret_cast<std::int32_t*>(d)[r] =
                        static_cast<std::int32_t>(c.i);
                    break;
            }
            return;
        case AggFunc::Avg:
            if (c.cnt == 0) {
                out.set_null(r);
                return;
            }
            reinterpret_cast<double*>(d)[r] =
                (is_float ? c.d : static_cast<double>(c.i)) /
                static_cast<double>(c.cnt);
            return;
    }
}

}  // namespace qe::ops::detail
