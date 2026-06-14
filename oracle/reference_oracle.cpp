//  WP-3 (seed of WP-9), extended at WP-5: independent reference oracle. See
//  reference_oracle.h.
//
//  The GROUP BY reference (run_reference_group_by) computes the grouped aggregate
//  DIRECTLY over the whole table with a std::map keyed by the canonical key tuple
//  — it shares NO code path with the engine's Aggregate operator / HashTable /
//  aggregate kernels, so "engine == reference" is a meaningful differential
//  (alongside the authoritative DuckDB diff). It reproduces the documented key
//  semantics independently: NULLs group together (kEqual), and F64 zero
//  canonicalizes to +0.0 so its output key matches the engine's read-back.
#include "oracle/reference_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "core/validity.h"
#include "expr/expr.h"
#include "ops/aggregate.h"

namespace qe::oracle {
namespace {

// Read cell at dense index `r` from a dense (unselected) evaluated column.
Cell read_dense(const OwnedColumn& oc, std::size_t r) {
    const Column c = oc.view();
    Cell cell;
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) {
        cell.is_null = true;
        return cell;
    }
    switch (c.type) {
        case Type::I32:
            cell.i = reinterpret_cast<const std::int32_t*>(c.data)[r];
            break;
        case Type::I64:
        case Type::TS:
            cell.i = reinterpret_cast<const std::int64_t*>(c.data)[r];
            break;
        case Type::BOOL:
            cell.i = reinterpret_cast<const std::uint8_t*>(c.data)[r];
            break;
        case Type::F64:
            cell.f = reinterpret_cast<const double*>(c.data)[r];
            break;
    }
    return cell;
}

// ---- GROUP BY reference ----------------------------------------------------

// A numeric value read from a table column at physical row r.
struct Num {
    bool valid = false;
    std::int64_t i = 0;
    double d = 0.0;
};

Num read_num(const Column& c, std::size_t r) {
    Num n;
    n.valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!n.valid) return n;
    switch (c.type) {
        case Type::I32:
            n.i = reinterpret_cast<const std::int32_t*>(c.data)[r];
            break;
        case Type::I64:
        case Type::TS:
            n.i = reinterpret_cast<const std::int64_t*>(c.data)[r];
            break;
        case Type::BOOL:
            n.i = reinterpret_cast<const std::uint8_t*>(c.data)[r];
            break;
        case Type::F64:
            n.d = reinterpret_cast<const double*>(c.data)[r];
            break;
    }
    return n;
}

// One key-column word for grouping: (is_null, canonical 64-bit pattern). NULLs
// compare equal (kEqual); F64 zero -> +0.0 bits to match the engine's read-back.
std::pair<bool, std::uint64_t> key_word(const Column& c, std::size_t r, Type t) {
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) return {true, 0};
    std::uint64_t w = 0;
    switch (t) {
        case Type::I32: {
            const auto v = reinterpret_cast<const std::int32_t*>(c.data)[r];
            w = static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
            break;
        }
        case Type::I64:
        case Type::TS: {
            const auto v = reinterpret_cast<const std::int64_t*>(c.data)[r];
            std::memcpy(&w, &v, 8);
            break;
        }
        case Type::BOOL:
            w = reinterpret_cast<const std::uint8_t*>(c.data)[r] ? 1u : 0u;
            break;
        case Type::F64: {
            double v = reinterpret_cast<const double*>(c.data)[r];
            if (v == 0.0) v = 0.0;  // collapse -0.0 to +0.0
            if (std::isnan(v)) {
                w = 0x7ff8000000000000ull;  // canonical quiet NaN
            } else {
                std::memcpy(&w, &v, 8);
            }
            break;
        }
    }
    return {false, w};
}

// Decode a (canonical) key word back into an output Cell of type t.
Cell decode_key(bool is_null, std::uint64_t w, Type t) {
    Cell c;
    if (is_null) {
        c.is_null = true;
        return c;
    }
    switch (t) {
        case Type::I32:
            c.i = static_cast<std::int32_t>(static_cast<std::uint32_t>(w));
            break;
        case Type::I64:
        case Type::TS: {
            std::int64_t v;
            std::memcpy(&v, &w, 8);
            c.i = v;
            break;
        }
        case Type::BOOL:
            c.i = static_cast<std::int64_t>(w & 1ull);
            break;
        case Type::F64: {
            double v;
            std::memcpy(&v, &w, 8);
            c.f = v;
            break;
        }
    }
    return c;
}

// Per-group, per-agg accumulator (independent of the engine's AggCell).
struct Acc {
    std::int64_t i = 0;
    double d = 0.0;
    std::int64_t cnt = 0;
};

Acc init_acc(AggFunc f) {
    Acc a;
    if (f == AggFunc::Min) {
        a.i = std::numeric_limits<std::int64_t>::max();
        a.d = std::numeric_limits<double>::infinity();
    } else if (f == AggFunc::Max) {
        a.i = std::numeric_limits<std::int64_t>::min();
        a.d = -std::numeric_limits<double>::infinity();
    }
    return a;
}

ResultSet run_reference_group_by(const Table& table, const LogicalQuery& q) {
    const GroupBy& gb = *q.group_by;
    const Schema out = query_output_schema(table.schema(), q);
    ResultSet rs;
    for (const auto& f : out.fields) rs.types.push_back(f.second);

    const Batch full = table.full_batch_view();
    const std::size_t n = table.num_rows();
    const bool global = gb.keys.empty();

    // Per-agg input type / float-ness.
    std::vector<Type> in_type(gb.aggs.size(), Type::I64);
    std::vector<bool> is_float(gb.aggs.size(), false);
    for (std::size_t a = 0; a < gb.aggs.size(); ++a) {
        if (gb.aggs[a].func != AggFunc::CountStar) {
            in_type[a] = table.schema().fields[gb.aggs[a].input_col].second;
            is_float[a] = in_type[a] == Type::F64;
        }
    }

    // Group accumulators, in first-seen order; keyed by canonical key tuple.
    using KeyTuple = std::vector<std::pair<bool, std::uint64_t>>;
    std::map<KeyTuple, std::size_t> index;
    std::vector<KeyTuple> group_keys;
    std::vector<std::vector<Acc>> group_acc;  // [group][agg]

    auto ensure_global = [&]() {
        if (group_acc.empty()) {
            group_keys.emplace_back();  // empty key tuple
            std::vector<Acc> accs(gb.aggs.size());
            for (std::size_t a = 0; a < gb.aggs.size(); ++a)
                accs[a] = init_acc(gb.aggs[a].func);
            group_acc.push_back(std::move(accs));
        }
    };
    if (global) ensure_global();  // exactly one row even over empty input

    // Filter mask (NULL predicate => row excluded). Skip evaluation on n==0
    // (expr::evaluate aborts on a 0-row batch — see WP-3 report).
    std::vector<char> keep(n, 1);
    if (n > 0 && q.has_filter()) {
        OwnedColumn pred = expr::evaluate(q.filter, full);
        const Column pv = pred.view();
        const auto* bits = reinterpret_cast<const std::uint8_t*>(pv.data);
        for (std::size_t r = 0; r < n; ++r) {
            const bool valid =
                pv.all_valid || validity::get_bit(pv.validity, r);
            keep[r] = (valid && bits[r] != 0) ? 1 : 0;
        }
    }

    for (std::size_t r = 0; r < n; ++r) {
        if (!keep[r]) continue;
        std::size_t g;
        if (global) {
            g = 0;
        } else {
            KeyTuple kt;
            kt.reserve(gb.keys.size());
            for (std::uint32_t kc : gb.keys)
                kt.push_back(key_word(full.cols[kc], r,
                                      table.schema().fields[kc].second));
            auto it = index.find(kt);
            if (it == index.end()) {
                g = group_acc.size();
                index.emplace(kt, g);
                group_keys.push_back(kt);
                std::vector<Acc> accs(gb.aggs.size());
                for (std::size_t a = 0; a < gb.aggs.size(); ++a)
                    accs[a] = init_acc(gb.aggs[a].func);
                group_acc.push_back(std::move(accs));
            } else {
                g = it->second;
            }
        }

        for (std::size_t a = 0; a < gb.aggs.size(); ++a) {
            const AggSpec& spec = gb.aggs[a];
            Acc& acc = group_acc[g][a];
            if (spec.func == AggFunc::CountStar) {
                ++acc.cnt;
                continue;
            }
            const Num v = read_num(full.cols[spec.input_col], r);
            switch (spec.func) {
                case AggFunc::Count:
                    if (v.valid) ++acc.cnt;
                    break;
                case AggFunc::Sum:
                case AggFunc::Avg:
                    if (v.valid) {
                        if (is_float[a])
                            acc.d += v.d;
                        else
                            acc.i += v.i;
                        ++acc.cnt;
                    }
                    break;
                case AggFunc::Min:
                    if (v.valid) {
                        if (is_float[a])
                            acc.d = std::min(acc.d, v.d);
                        else
                            acc.i = std::min(acc.i, v.i);
                        ++acc.cnt;
                    }
                    break;
                case AggFunc::Max:
                    if (v.valid) {
                        if (is_float[a])
                            acc.d = std::max(acc.d, v.d);
                        else
                            acc.i = std::max(acc.i, v.i);
                        ++acc.cnt;
                    }
                    break;
                case AggFunc::CountStar:
                    break;  // handled above
            }
        }
    }

    // Emit one ResultSet row per group: key cells then aggregate cells.
    for (std::size_t g = 0; g < group_acc.size(); ++g) {
        std::vector<Cell> row;
        row.reserve(gb.keys.size() + gb.aggs.size());
        for (std::size_t j = 0; j < gb.keys.size(); ++j) {
            const Type kt = table.schema().fields[gb.keys[j]].second;
            row.push_back(
                decode_key(group_keys[g][j].first, group_keys[g][j].second, kt));
        }
        for (std::size_t a = 0; a < gb.aggs.size(); ++a) {
            const AggSpec& spec = gb.aggs[a];
            const Acc& acc = group_acc[g][a];
            const Type rt = agg_result_type(spec.func, in_type[a]);
            Cell c;
            switch (spec.func) {
                case AggFunc::CountStar:
                case AggFunc::Count:
                    c.i = acc.cnt;
                    break;
                case AggFunc::Sum:
                    if (acc.cnt == 0)
                        c.is_null = true;
                    else if (rt == Type::F64)
                        c.f = acc.d;
                    else
                        c.i = acc.i;
                    break;
                case AggFunc::Min:
                case AggFunc::Max:
                    if (acc.cnt == 0)
                        c.is_null = true;
                    else if (rt == Type::F64)
                        c.f = acc.d;
                    else
                        c.i = acc.i;
                    break;
                case AggFunc::Avg:
                    if (acc.cnt == 0)
                        c.is_null = true;
                    else
                        c.f = (is_float[a] ? acc.d
                                           : static_cast<double>(acc.i)) /
                              static_cast<double>(acc.cnt);
                    break;
            }
            row.push_back(c);
        }
        rs.rows.push_back(std::move(row));
    }
    return rs;
}

}  // namespace

ResultSet run_reference(const Table& table, const LogicalQuery& q) {
    if (q.has_group_by()) return run_reference_group_by(table, q);

    const Batch full = table.full_batch_view();
    const std::size_t n = table.num_rows();

    // 0. Empty table: result is trivially empty. (Also a guard: expr::evaluate()
    //    -> compact_column aborts on a row_count==0 batch — see WP-3 report.
    //    Like the engine, which short-circuits a 0-row scan, we never evaluate.)
    if (n == 0) {
        ResultSet rs;
        for (const auto& p : q.projections) rs.types.push_back(p.expr.type());
        return rs;
    }

    // 1. Which rows pass the filter (default: all rows). NULL predicate => no.
    std::vector<char> keep(n, 1);
    if (q.has_filter()) {
        OwnedColumn pred = expr::evaluate(q.filter, full);
        const Column pv = pred.view();
        const auto* bits = reinterpret_cast<const std::uint8_t*>(pv.data);
        for (std::size_t r = 0; r < n; ++r) {
            const bool valid =
                pv.all_valid || validity::get_bit(pv.validity, r);
            keep[r] = (valid && bits[r] != 0) ? 1 : 0;
        }
    }

    // 2. Evaluate each projection over the whole table (dense, length n).
    ResultSet rs;
    std::vector<OwnedColumn> proj;
    proj.reserve(q.projections.size());
    for (const auto& p : q.projections) {
        proj.push_back(expr::evaluate(p.expr, full));
        rs.types.push_back(p.expr.type());
    }

    // 3. Emit one ResultSet row per kept table row.
    for (std::size_t r = 0; r < n; ++r) {
        if (!keep[r]) continue;
        std::vector<Cell> row;
        row.reserve(proj.size());
        for (const auto& pc : proj) row.push_back(read_dense(pc, r));
        rs.rows.push_back(std::move(row));
    }
    return rs;
}

}  // namespace qe::oracle
