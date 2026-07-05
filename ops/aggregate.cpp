//  WP-5: Hash aggregation / GROUP BY implementation. See ops/aggregate.h.
//
//  GROUPING uses the frozen HashTable (ops/hashtable.h) with NullPolicy::kEqual
//  (SQL GROUP BY groups all NULLs together). insert_or_find hands each row a
//  stable, dense group id; per-group aggregate state is indexed by that id (ids
//  never change across table growth — the table guarantees it).
//
//  TWO ACCUMULATION PATHS:
//   * GLOBAL (zero keys): a pure reduction over the whole input. It runs through
//     the SIMD masked-reduction kernels (ops/agg_kernels.h) — vector or scalar
//     per AggKernelPath — which is where the scalar==vector gate bites.
//   * GROUPED (>=1 key): an inherently-sequential scatter into per-group state
//     (duplicate group ids within a SIMD lane cannot be scatter-added safely or
//     portably with Highway), so it is scalar control flow, exactly like the
//     WP-4 probe walk. Its correctness is pinned by the DuckDB differential and
//     the mutation self-test, not by scalar==vector. (See the WP report.)
//
//  PIPELINE-BREAKER: open() drains the child fully and builds all state; next()
//  then yields the groups in dense <=kOutBatch batches (keys then aggregates).
//  Output materialization builds OwnedColumns directly and never routes through
//  compact_column, so the known 0-row compact_column abort is not reachable from
//  here (the global-over-empty single row and the grouped-empty zero rows are
//  both produced without it).

#include "ops/aggregate.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "core/string_dict.h"
#include "ops/agg_internal.h"
#include "ops/agg_kernels.h"

namespace qe {

namespace detail = qe::ops::detail;

struct Aggregate::State {
    // grouped path
    std::unique_ptr<HashTable> ht;
    std::vector<std::vector<detail::AggCell>> cells_grp;  // [agg][group]
    std::size_t num_groups = 0;

    // global path (zero keys): one cell per agg, exactly one output row
    std::vector<detail::AggCell> cells_global;

    // shared, computed at open()
    std::vector<Type> key_types;     // child types of the key columns, key order
    std::vector<Type> input_types;   // [agg]; I64 placeholder for CountStar
    std::vector<bool> is_float;      // [agg]

    // scratch reused across batches by the global kernel path
    std::vector<std::int64_t> scratch_i;
    std::vector<double> scratch_d;

    // WP-7b: the aggregate's OWNED canonical string dict. STR group keys and
    // MIN/MAX(str) results are interned here (by string VALUE) during the drain so
    // that (a) equal strings from any source dict share one code => the frozen
    // HashTable groups them together, and (b) the codes the output columns publish
    // stay valid AFTER the child (and its dicts) are closed at end-of-drain.
    std::shared_ptr<StringDict> str_dict;
};

Aggregate::Aggregate(std::unique_ptr<Operator> child,
                     std::vector<std::uint32_t> key_cols,
                     std::vector<AggSpec> aggs)
    : child_(std::move(child)),
      key_cols_(std::move(key_cols)),
      aggs_(std::move(aggs)) {
    assert(!aggs_.empty() && "Aggregate needs >=1 aggregate");
    child_schema_ = child_->output_schema();
}

Type agg_result_type(AggFunc func, Type input) {
    switch (func) {
        case AggFunc::CountStar:
        case AggFunc::Count:
            return Type::I64;
        case AggFunc::Sum:
            // WP-7b: SUM(str)/AVG(str) are type errors (DuckDB rejects them too).
            if (input == Type::STR)
                throw std::invalid_argument("SUM is not defined on STR (VARCHAR)");
            return input == Type::F64 ? Type::F64 : Type::I64;  // SUM(int)->I64
        case AggFunc::Avg:
            if (input == Type::STR)
                throw std::invalid_argument("AVG is not defined on STR (VARCHAR)");
            return Type::F64;
        case AggFunc::Min:
        case AggFunc::Max:
            return input;  // same type as the input column (STR -> STR by VALUE)
    }
    return Type::I64;  // unreachable
}

Schema Aggregate::output_schema() const {
    Schema s;
    s.fields.reserve(key_cols_.size() + aggs_.size());
    for (std::uint32_t kc : key_cols_)
        s.fields.emplace_back(child_schema_.fields[kc].first,
                              child_schema_.fields[kc].second);
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        s.fields.emplace_back(a.out_name, agg_result_type(a.func, in));
    }
    return s;
}

namespace {

// Fold one row's value into a per-group cell (the GROUPED scatter step). This is
// the CORRECT reference; the mutant copies it with one defect.
inline void scatter_row(detail::AggCell& cell, AggFunc func, bool is_float,
                        const detail::ColVal& v) {
    switch (func) {
        case AggFunc::CountStar:
            ++cell.cnt;  // counts the row regardless of any value's nullness
            return;
        case AggFunc::Count:
            if (v.valid) ++cell.cnt;
            return;
        case AggFunc::Sum:
        case AggFunc::Avg:
            if (!v.valid) return;  // ignore NULLs (do NOT fold as 0/count them)
            if (is_float)
                cell.d += v.d;
            else
                cell.i += v.i;
            ++cell.cnt;
            return;
        case AggFunc::Min:
            if (!v.valid) return;
            if (is_float)
                // NaN-greatest total order (agg_internal.h): NaN wins only when
                // the whole group is NaN. Raw std::min silently dropped NaNs.
                cell.d = detail::f64_min_total(cell.d, v.d);
            else
                cell.i = std::min(cell.i, v.i);
            ++cell.cnt;
            return;
        case AggFunc::Max:
            if (!v.valid) return;
            if (is_float)
                cell.d = detail::f64_max_total(cell.d, v.d);
            else
                cell.i = std::max(cell.i, v.i);
            ++cell.cnt;
            return;
    }
}

// WP-7b: canonicalize one STR key column into `dst` (the aggregate's owned dict),
// preserving the source's PHYSICAL layout and validity (so the batch's selection
// vector still indexes it correctly). Equal string VALUES — even from different
// source dicts — get the same code in `dst`, so the frozen HashTable (which hashes
// the raw code via normalize_value) groups them together, and the group-key
// read-back is a code valid in `dst` after the child closes. Grouping by raw,
// un-canonicalized codes would be the "by code not by value" hazard.
OwnedColumn canonicalize_str_key(const Column& src, const SelectionVector* sel,
                                 StringDict& dst) {
    const std::size_t len = src.len;
    OwnedColumn out = OwnedColumn::make(Type::STR, len);
    auto* codes = reinterpret_cast<std::int32_t*>(out.mutable_data());
    const auto* src_codes = reinterpret_cast<const std::int32_t*>(src.data);
    // Audit H3: touch ONLY the batch's live rows. Unselected physical slots are
    // not required to hold meaningful codes (the Batch contract), so resolving
    // them through the dict would be UB — and interning them pollutes the
    // aggregate's owned dict with filtered-out values. The physical layout is
    // preserved (sel keeps indexing the output); dead slots get code 0 and are
    // never read by any sel-aware consumer.
    std::memset(codes, 0, len * sizeof(std::int32_t));
    const std::size_t n = sel ? sel->len : len;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = sel_at(sel, k);
        const bool valid = src.all_valid || validity::get_bit(src.validity, p);
        if (valid)
            codes[p] = dst.intern(src.dict->at(src_codes[p]));
        else
            out.set_null(p);
    }
    return out;
}

// WP-7b: fold one STR value `s` into a per-group MIN/MAX cell. The running extremum
// is a code in the aggregate's owned dict `dst`, chosen by lexicographic byte VALUE
// (std::string_view ordering == DuckDB VARCHAR ordering for ASCII), never by code.
inline void scatter_str_minmax(detail::AggCell& cell, AggFunc func,
                               std::string_view s, StringDict& dst) {
    if (cell.cnt == 0) {
        cell.i = dst.intern(s);
    } else {
        const std::string_view cur = dst.at(static_cast<std::int32_t>(cell.i));
        const bool take = (func == AggFunc::Min) ? (s < cur) : (s > cur);
        if (take) cell.i = dst.intern(s);
    }
    ++cell.cnt;
}

}  // namespace

void Aggregate::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    State& st = *state_;
    st.str_dict = std::make_shared<StringDict>();  // WP-7b owned canonical dict

    // Resolve per-agg input types / float-ness and the key types.
    st.input_types.reserve(aggs_.size());
    st.is_float.reserve(aggs_.size());
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        st.input_types.push_back(in);
        st.is_float.push_back(in == Type::F64);
    }
    for (std::uint32_t kc : key_cols_)
        st.key_types.push_back(child_schema_.fields[kc].second);

    const bool global = key_cols_.empty();
    if (global) {
        st.cells_global.resize(aggs_.size());
        for (std::size_t a = 0; a < aggs_.size(); ++a)
            st.cells_global[a] =
                detail::init_agg_cell(aggs_[a].func, st.is_float[a]);
    } else {
        st.ht = std::make_unique<HashTable>(st.key_types, NullPolicy::kEqual);
        st.cells_grp.resize(aggs_.size());
    }

    drain_and_build();

    emit_cursor_ = 0;
    emitted_empty_global_ = false;
}

void Aggregate::drain_and_build() {
    State& st = *state_;
    const bool global = key_cols_.empty();

    child_->open();

    std::vector<Column> key_views;        // reused per batch (grouped)
    std::vector<std::uint32_t> gids;      // reused per batch (grouped)
    std::vector<OwnedColumn> canon_keys;  // WP-7b: per-batch canonical STR keys

    while (std::optional<Batch> in = child_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;  // contract: batches are non-empty, but be total

        if (global) {
            // --- GLOBAL reduction via the SIMD masked-reduction kernels -------
            for (std::size_t a = 0; a < aggs_.size(); ++a) {
                const AggSpec& spec = aggs_[a];
                detail::AggCell& cell = st.cells_global[a];
                const bool isf = st.is_float[a];

                if (spec.func == AggFunc::CountStar) {
                    cell.cnt += static_cast<std::int64_t>(n);
                    continue;
                }
                const Column& col = b.cols[spec.input_col];

                if (spec.func == AggFunc::Count) {
                    std::int64_t nn = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        if (detail::read_col(col, b.sel, k).valid) ++nn;
                    cell.cnt += nn;
                    continue;
                }

                if (col.type == Type::STR) {
                    // WP-7b: global MIN/MAX(str) — by VALUE into the owned dict.
                    // (SUM/AVG(str) were rejected by agg_result_type.)
                    for (std::size_t k = 0; k < n; ++k) {
                        const detail::ColVal v = detail::read_col(col, b.sel, k);
                        if (!v.valid) continue;
                        scatter_str_minmax(
                            cell, spec.func,
                            col.dict->at(static_cast<std::int32_t>(v.i)),
                            *st.str_dict);
                    }
                    continue;
                }

                // SUM / AVG / MIN / MAX: build the identity-folded array + count.
                std::int64_t nonnull = 0;
                const AggFunc f = spec.func;
                if (isf) {
                    st.scratch_d.resize(n);
                    // MIN's identity is NaN under the NaN-greatest total order
                    // (agg_internal.h): NULL slots folded as NaN are ignored by
                    // the total-order reduction unless the batch has no real
                    // value at all — which the nonnull count already handles.
                    const double id = (f == AggFunc::Min)
                                          ? std::numeric_limits<double>::quiet_NaN()
                                      : (f == AggFunc::Max)
                                          ? -std::numeric_limits<double>::infinity()
                                          : 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        const detail::ColVal v = detail::read_col(col, b.sel, k);
                        if (v.valid) {
                            st.scratch_d[k] = v.d;
                            ++nonnull;
                        } else {
                            st.scratch_d[k] = id;
                        }
                    }
                    const double* p = st.scratch_d.data();
                    if (f == AggFunc::Sum || f == AggFunc::Avg) {
                        cell.d += (kernel_path_ == AggKernelPath::kVector)
                                      ? ops::agg_sum_f64_vec(p, n)
                                      : ops::agg_sum_f64_scalar(p, n);
                    } else if (f == AggFunc::Min) {
                        const double m = (kernel_path_ == AggKernelPath::kVector)
                                             ? ops::agg_min_f64_vec(p, n)
                                             : ops::agg_min_f64_scalar(p, n);
                        // Cross-batch fold under the same NaN-greatest total
                        // order as the kernels (raw std::min drops NaN).
                        cell.d = detail::f64_min_total(cell.d, m);
                    } else {  // Max
                        const double m = (kernel_path_ == AggKernelPath::kVector)
                                             ? ops::agg_max_f64_vec(p, n)
                                             : ops::agg_max_f64_scalar(p, n);
                        cell.d = detail::f64_max_total(cell.d, m);
                    }
                } else {
                    st.scratch_i.resize(n);
                    const std::int64_t id =
                        (f == AggFunc::Min)
                            ? std::numeric_limits<std::int64_t>::max()
                        : (f == AggFunc::Max)
                            ? std::numeric_limits<std::int64_t>::min()
                            : 0;
                    for (std::size_t k = 0; k < n; ++k) {
                        const detail::ColVal v = detail::read_col(col, b.sel, k);
                        if (v.valid) {
                            st.scratch_i[k] = v.i;
                            ++nonnull;
                        } else {
                            st.scratch_i[k] = id;
                        }
                    }
                    const std::int64_t* p = st.scratch_i.data();
                    if (f == AggFunc::Sum || f == AggFunc::Avg) {
                        cell.i += (kernel_path_ == AggKernelPath::kVector)
                                      ? ops::agg_sum_i64_vec(p, n)
                                      : ops::agg_sum_i64_scalar(p, n);
                    } else if (f == AggFunc::Min) {
                        const std::int64_t m =
                            (kernel_path_ == AggKernelPath::kVector)
                                ? ops::agg_min_i64_vec(p, n)
                                : ops::agg_min_i64_scalar(p, n);
                        cell.i = std::min(cell.i, m);
                    } else {  // Max
                        const std::int64_t m =
                            (kernel_path_ == AggKernelPath::kVector)
                                ? ops::agg_max_i64_vec(p, n)
                                : ops::agg_max_i64_scalar(p, n);
                        cell.i = std::max(cell.i, m);
                    }
                }
                cell.cnt += nonnull;
            }
            continue;
        }

        // --- GROUPED scatter ----------------------------------------------------
        // WP-7b: STR key columns are canonicalized into the owned dict by VALUE
        // before the frozen HashTable sees them; non-STR keys pass through. canon
        // is reserved (no reallocation) so the views below stay valid for the call.
        key_views.clear();
        key_views.reserve(key_cols_.size());
        canon_keys.clear();
        canon_keys.reserve(key_cols_.size());
        for (std::uint32_t col_idx : key_cols_) {
            const Column& kcol = b.cols[col_idx];
            if (kcol.type == Type::STR) {
                canon_keys.push_back(canonicalize_str_key(kcol, b.sel, *st.str_dict));
                key_views.push_back(canon_keys.back().view());
            } else {
                key_views.push_back(kcol);
            }
        }
        const KeyColumns kc{key_views.data(), key_views.size(), b.sel};

        gids.resize(n);
        st.ht->insert_or_find(kc, n, gids.data(), hash_path_);

        const std::size_t G = st.ht->num_groups();
        if (G > st.num_groups) {
            for (std::size_t a = 0; a < aggs_.size(); ++a) {
                st.cells_grp[a].resize(G);
                for (std::size_t g = st.num_groups; g < G; ++g)
                    st.cells_grp[a][g] =
                        detail::init_agg_cell(aggs_[a].func, st.is_float[a]);
            }
            st.num_groups = G;
        }

        for (std::size_t a = 0; a < aggs_.size(); ++a) {
            const AggSpec& spec = aggs_[a];
            const bool isf = st.is_float[a];
            auto& cells = st.cells_grp[a];
            if (spec.func == AggFunc::CountStar) {
                for (std::size_t k = 0; k < n; ++k) ++cells[gids[k]].cnt;
                continue;
            }
            const Column& col = b.cols[spec.input_col];
            const bool str_minmax =
                col.type == Type::STR &&
                (spec.func == AggFunc::Min || spec.func == AggFunc::Max);
            for (std::size_t k = 0; k < n; ++k) {
                const detail::ColVal v = detail::read_col(col, b.sel, k);
                if (str_minmax) {
                    // WP-7b: MIN/MAX(str) compare by VALUE via the source dict.
                    if (!v.valid) continue;
                    scatter_str_minmax(
                        cells[gids[k]], spec.func,
                        col.dict->at(static_cast<std::int32_t>(v.i)),
                        *st.str_dict);
                } else {
                    scatter_row(cells[gids[k]], spec.func, isf, v);
                }
            }
        }
    }

    child_->close();
}

std::optional<Batch> Aggregate::next() {
    assert(opened_ && "next() before open()");
    State& st = *state_;
    const bool global = key_cols_.empty();

    if (global) {
        if (emitted_empty_global_) return std::nullopt;  // exactly one row
        emitted_empty_global_ = true;
        OwnedBatch out;
        for (std::size_t a = 0; a < aggs_.size(); ++a) {
            const Type rt = agg_result_type(aggs_[a].func, st.input_types[a]);
            OwnedColumn c = OwnedColumn::make(rt, 1);
            detail::write_agg_cell(c, 0, aggs_[a].func, st.is_float[a], rt,
                                   st.cells_global[a]);
            if (rt == Type::STR) c.set_dict(st.str_dict);  // WP-7b MIN/MAX(str)
            out.add_column(std::move(c));
        }
        current_ = std::move(out);
        return current_.view();
    }

    // GROUPED: emit groups [emit_cursor_, emit_cursor_+m) as one dense batch.
    const std::size_t G = st.num_groups;
    if (emit_cursor_ >= G) return std::nullopt;  // empty input => zero rows
    const std::size_t m = std::min(kOutBatch, G - emit_cursor_);

    OwnedBatch out;
    // Key columns (in key order), materialized from the table's key read-back.
    for (std::size_t j = 0; j < key_cols_.size(); ++j) {
        const Type kt = st.key_types[j];
        OwnedColumn c = OwnedColumn::make(kt, m);
        for (std::size_t r = 0; r < m; ++r) {
            const std::uint32_t g =
                static_cast<std::uint32_t>(emit_cursor_ + r);
            detail::write_key_cell(c, r, kt, st.ht->group_is_null(g, j),
                                   st.ht->group_key_word(g, j));
        }
        if (kt == Type::STR) c.set_dict(st.str_dict);  // WP-7b STR group key
        out.add_column(std::move(c));
    }
    // Aggregate result columns.
    for (std::size_t a = 0; a < aggs_.size(); ++a) {
        const Type rt = agg_result_type(aggs_[a].func, st.input_types[a]);
        OwnedColumn c = OwnedColumn::make(rt, m);
        for (std::size_t r = 0; r < m; ++r)
            detail::write_agg_cell(c, r, aggs_[a].func, st.is_float[a], rt,
                                   st.cells_grp[a][emit_cursor_ + r]);
        if (rt == Type::STR) c.set_dict(st.str_dict);  // WP-7b MIN/MAX(str)
        out.add_column(std::move(c));
    }

    emit_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void Aggregate::close() {
    // Child already closed at end-of-drain; release built state.
    state_.reset();
    opened_ = false;
}

}  // namespace qe
