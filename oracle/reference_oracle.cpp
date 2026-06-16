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
#include <optional>
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

// WP-7: independently sort a ResultSet's rows by an ORDER BY key list, for the
// POSITIONAL differential. Shares NO code with the engine's radix/comparison sort
// (it compares already-materialized Cells), so "engine == reference" under ORDERED
// compare is a meaningful check of the sort. std::stable_sort matches the engine's
// stability on the child's (table/first-seen) row order; NULL placement is
// ABSOLUTE (independent of ASC/DESC), and F64 uses IEEE order — the same contract
// the engine's comparator and DuckDB obey.
void apply_order_by(ResultSet& rs, const std::vector<SortKey>& keys) {
    std::stable_sort(
        rs.rows.begin(), rs.rows.end(),
        [&](const std::vector<Cell>& a, const std::vector<Cell>& b) {
            for (const SortKey& k : keys) {
                const Cell& ca = a[k.col];
                const Cell& cb = b[k.col];
                if (ca.is_null || cb.is_null) {
                    if (ca.is_null && cb.is_null) continue;
                    const bool a_null = ca.is_null;
                    return (k.nulls == NullOrder::First) ? a_null : !a_null;
                }
                int cmp;
                if (rs.types[k.col] == Type::F64)
                    cmp = (ca.f < cb.f) ? -1 : (ca.f > cb.f) ? 1 : 0;
                else
                    cmp = (ca.i < cb.i) ? -1 : (ca.i > cb.i) ? 1 : 0;
                if (cmp != 0)
                    return (k.dir == SortDir::Asc) ? (cmp < 0) : (cmp > 0);
            }
            return false;  // fully equal: stable_sort keeps original order
        });
}

}  // namespace

ResultSet run_reference(const Table& table, const LogicalQuery& q) {
    if (q.has_group_by()) {
        ResultSet rs = run_reference_group_by(table, q);
        if (q.has_order_by()) apply_order_by(rs, *q.order_by);
        return rs;
    }

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
    if (q.has_order_by()) apply_order_by(rs, *q.order_by);  // WP-7
    return rs;
}

namespace {

// Read one cell from a (dense, whole-table) Column view at physical row r.
Cell read_cell_col(const Column& c, std::size_t r) {
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

}  // namespace

ResultSet run_join_reference(const Table& probe, const Table& build,
                             const JoinQuery& jq) {
    const Schema out = join_output_schema(probe.schema(), build.schema());
    ResultSet rs;
    for (const auto& f : out.fields) rs.types.push_back(f.second);

    const Batch pb = probe.full_batch_view();
    const Batch bb = build.full_batch_view();
    const std::size_t np = probe.num_rows();
    const std::size_t nb = build.num_rows();
    const std::size_t nk = jq.build_keys.size();

    // Canonical key tuple of a row, or std::nullopt if ANY key column is NULL
    // (kNeverMatch: a NULL key never matches, not even another NULL).
    using KeyTuple = std::vector<std::uint64_t>;
    auto key_of = [&](const Batch& b, const std::vector<std::uint32_t>& keys,
                      const Schema& sch, std::size_t r)
        -> std::optional<KeyTuple> {
        KeyTuple kt;
        kt.reserve(nk);
        for (std::size_t j = 0; j < nk; ++j) {
            const std::uint32_t kc = keys[j];
            const auto [is_null, w] =
                key_word(b.cols[kc], r, sch.fields[kc].second);
            if (is_null) return std::nullopt;  // dead key, never matches
            kt.push_back(w);
        }
        return kt;
    };

    // Build index: canonical build-key tuple -> list of build row indices. Only
    // non-null-key build rows are inserted (null keys never match).
    std::map<KeyTuple, std::vector<std::size_t>> index;
    for (std::size_t r = 0; r < nb; ++r) {
        auto kt = key_of(bb, jq.build_keys, build.schema(), r);
        if (kt) index[*kt].push_back(r);
    }

    auto emit_build_null = [&](std::vector<Cell>& row) {
        for (std::size_t c = 0; c < build.schema().fields.size(); ++c) {
            Cell cell;
            cell.is_null = true;
            row.push_back(cell);
        }
    };

    for (std::size_t r = 0; r < np; ++r) {
        std::vector<Cell> probe_cells;
        probe_cells.reserve(probe.schema().fields.size());
        for (std::size_t c = 0; c < probe.schema().fields.size(); ++c)
            probe_cells.push_back(read_cell_col(pb.cols[c], r));

        auto kt = key_of(pb, jq.probe_keys, probe.schema(), r);
        const std::vector<std::size_t>* matches = nullptr;
        if (kt) {
            auto it = index.find(*kt);
            if (it != index.end()) matches = &it->second;
        }

        if (matches) {
            for (std::size_t br : *matches) {
                std::vector<Cell> row = probe_cells;
                for (std::size_t c = 0; c < build.schema().fields.size(); ++c)
                    row.push_back(read_cell_col(bb.cols[c], br));
                rs.rows.push_back(std::move(row));
            }
        } else if (jq.type == JoinType::Left) {
            std::vector<Cell> row = probe_cells;
            emit_build_null(row);
            rs.rows.push_back(std::move(row));
        }
    }
    return rs;
}

// ---- WP-12: backward as-of join reference ----------------------------------
namespace {

// A timestamp read as int64 (the comparison domain), independent of the engine:
// I32 widened, I64/TS verbatim. valid==false for a NULL timestamp.
struct RefTime {
    bool valid = false;
    std::int64_t t = 0;
};
RefTime read_time_ref(const Column& c, std::size_t r) {
    RefTime out;
    out.valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!out.valid) return out;
    switch (c.type) {
        case Type::I32:
            out.t = reinterpret_cast<const std::int32_t*>(c.data)[r];
            break;
        case Type::I64:
        case Type::TS:
            out.t = reinterpret_cast<const std::int64_t*>(c.data)[r];
            break;
        case Type::F64:
        case Type::BOOL:
            out.valid = false;  // not a valid timestamp type
            break;
    }
    return out;
}

}  // namespace

ResultSet run_asof_reference(const Table& probe, const Table& build,
                             const std::vector<std::uint32_t>& left_keys,
                             const std::vector<std::uint32_t>& right_keys,
                             std::uint32_t left_time, std::uint32_t right_time,
                             qe::tsx::AsofType type,
                             std::optional<std::int64_t> tolerance) {
    Schema out = join_output_schema(probe.schema(), build.schema());
    ResultSet rs;
    for (const auto& f : out.fields) rs.types.push_back(f.second);

    const Batch pb = probe.full_batch_view();
    const Batch bb = build.full_batch_view();
    const std::size_t np = probe.num_rows();
    const std::size_t nb = build.num_rows();
    const std::size_t nk = right_keys.size();

    // Canonical key tuple of a row, or nullopt if ANY key column is NULL (a NULL
    // key never matches — same canonicalization as group_by/join above).
    using KeyTuple = std::vector<std::uint64_t>;
    auto key_of = [&](const Batch& b, const std::vector<std::uint32_t>& keys,
                      const Schema& sch,
                      std::size_t r) -> std::optional<KeyTuple> {
        KeyTuple kt;
        kt.reserve(nk);
        for (std::size_t j = 0; j < nk; ++j) {
            const std::uint32_t kc = keys[j];
            const auto [is_null, w] = key_word(b.cols[kc], r, sch.fields[kc].second);
            if (is_null) return std::nullopt;
            kt.push_back(w);
        }
        return kt;
    };

    auto emit_build_null = [&](std::vector<Cell>& row) {
        for (std::size_t c = 0; c < build.schema().fields.size(); ++c) {
            Cell cell;
            cell.is_null = true;
            row.push_back(cell);
        }
    };

    for (std::size_t r = 0; r < np; ++r) {
        std::vector<Cell> probe_cells;
        probe_cells.reserve(probe.schema().fields.size());
        for (std::size_t c = 0; c < probe.schema().fields.size(); ++c)
            probe_cells.push_back(read_cell_col(pb.cols[c], r));

        const auto pkey = key_of(pb, left_keys, probe.schema(), r);
        const RefTime pt = read_time_ref(pb.cols[left_time], r);

        // Brute-force scan for the nearest PRECEDING build row with an equal key.
        bool have = false;
        std::size_t best = 0;
        std::int64_t best_ts = 0;
        if (pkey && pt.valid) {
            for (std::size_t s = 0; s < nb; ++s) {
                const auto bkey = key_of(bb, right_keys, build.schema(), s);
                if (!bkey || *bkey != *pkey) continue;
                const RefTime bt = read_time_ref(bb.cols[right_time], s);
                if (!bt.valid || bt.t > pt.t) continue;  // `<=` boundary
                if (!have || bt.t > best_ts) {            // greatest tb <= t
                    have = true;
                    best = s;
                    best_ts = bt.t;
                }
            }
            if (have && tolerance && (pt.t - best_ts) > *tolerance)
                have = false;  // nearest is outside the backward window
        }

        if (have) {
            std::vector<Cell> row = probe_cells;
            for (std::size_t c = 0; c < build.schema().fields.size(); ++c)
                row.push_back(read_cell_col(bb.cols[c], best));
            rs.rows.push_back(std::move(row));
        } else if (type == qe::tsx::AsofType::Left) {
            std::vector<Cell> row = probe_cells;
            emit_build_null(row);
            rs.rows.push_back(std::move(row));
        }
        // INNER + no match: emit nothing.
    }
    return rs;
}

// ---- WP-8: plan reference interpreter --------------------------------------
namespace {

// Materialize a ResultSet back into a Table of `schema` (the dense, owned form the
// per-node references consume). Independent of the engine; pure cell copying.
Table rs_to_table(const ResultSet& rs, const Schema& schema) {
    const std::size_t n = rs.num_rows();
    std::vector<OwnedColumn> cols;
    cols.reserve(schema.fields.size());
    for (std::size_t c = 0; c < schema.fields.size(); ++c) {
        const Type t = schema.fields[c].second;
        OwnedColumn oc = OwnedColumn::make(t, n);
        std::byte* d = oc.mutable_data();
        for (std::size_t r = 0; r < n; ++r) {
            const Cell& cell = rs.rows[r][c];
            if (cell.is_null) {
                oc.set_null(r);
                continue;
            }
            switch (t) {
                case Type::I32:
                    reinterpret_cast<std::int32_t*>(d)[r] =
                        static_cast<std::int32_t>(cell.i);
                    break;
                case Type::I64:
                case Type::TS:
                    reinterpret_cast<std::int64_t*>(d)[r] = cell.i;
                    break;
                case Type::BOOL:
                    reinterpret_cast<std::uint8_t*>(d)[r] =
                        static_cast<std::uint8_t>(cell.i ? 1 : 0);
                    break;
                case Type::F64:
                    reinterpret_cast<double*>(d)[r] = cell.f;
                    break;
            }
        }
        cols.push_back(std::move(oc));
    }
    return Table(schema, std::move(cols));
}

// The identity SELECT (every column as-is) over `schema` — used to express Scan,
// Filter, and Sort as a run_reference() call (which always projects).
std::vector<Projection> identity_projections(const Schema& schema) {
    std::vector<Projection> ps;
    ps.reserve(schema.fields.size());
    for (std::uint32_t i = 0; i < schema.fields.size(); ++i)
        ps.push_back(Projection{"c" + std::to_string(i),
                                expr::col(schema.fields[i].second, i)});
    return ps;
}

using qe::plan::Plan;
using qe::plan::PlanKind;
using qe::plan::PlanNode;

// Evaluate a plan node to a ResultSet by composing the existing references over a
// materialized child Table (or two, for Join).
ResultSet eval_rs(const Plan& p) {
    const PlanNode& n = p.node();
    switch (n.kind) {
        case PlanKind::Scan: {
            LogicalQuery q;
            q.projections = identity_projections(n.table->schema());
            return run_reference(*n.table, q);
        }
        case PlanKind::Filter: {
            const Plan& c = n.children[0];
            const Table ct = rs_to_table(eval_rs(c), c.output_schema());
            LogicalQuery q;
            q.filter = n.predicate;
            q.projections = identity_projections(ct.schema());
            return run_reference(ct, q);
        }
        case PlanKind::Project: {
            const Plan& c = n.children[0];
            const Table ct = rs_to_table(eval_rs(c), c.output_schema());
            LogicalQuery q;
            q.projections = n.projections;
            return run_reference(ct, q);
        }
        case PlanKind::Aggregate: {
            const Plan& c = n.children[0];
            const Table ct = rs_to_table(eval_rs(c), c.output_schema());
            LogicalQuery q;
            q.group_by = GroupBy{n.group_keys, n.aggs};
            return run_reference(ct, q);
        }
        case PlanKind::Join: {
            const Plan& l = n.children[0];
            const Plan& r = n.children[1];
            const Table lt = rs_to_table(eval_rs(l), l.output_schema());
            const Table rt = rs_to_table(eval_rs(r), r.output_schema());
            JoinQuery jq{n.left_keys, n.right_keys, n.join_type};
            return run_join_reference(lt, rt, jq);
        }
        case PlanKind::Sort: {
            const Plan& c = n.children[0];
            const Table ct = rs_to_table(eval_rs(c), c.output_schema());
            LogicalQuery q;
            q.projections = identity_projections(ct.schema());
            q.order_by = n.sort_keys;
            return run_reference(ct, q);
        }
        case PlanKind::AsofJoin: {  // WP-12 (additive arm)
            const Plan& l = n.children[0];
            const Plan& r = n.children[1];
            const Table lt = rs_to_table(eval_rs(l), l.output_schema());
            const Table rt = rs_to_table(eval_rs(r), r.output_schema());
            return run_asof_reference(lt, rt, n.asof_left_keys, n.asof_right_keys,
                                      n.asof_left_time, n.asof_right_time,
                                      n.asof_type, n.asof_tolerance);
        }
    }
    return {};  // unreachable
}

}  // namespace

ResultSet run_plan_reference(const qe::plan::Plan& p) { return eval_rs(p); }

}  // namespace qe::oracle
