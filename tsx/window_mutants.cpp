//  WP-13 mutation self-test support. See tsx/window_mutants.h. A faithful copy of
//  tsx/window.cpp's tumbling group/scatter and sliding sort/sweep/emit loops reusing
//  the SAME ops/join_internal.h gather + ops/agg_internal.h finalizer + frozen
//  Sort/HashTable, differing by exactly one planted defect per mutation:
//   * kBucketEdge    — the tumbling bucket lower edge is bucket_id = (t + 1) / W.
//   * kFrameOffByOne — the sliding frame reaches P+1 preceding rows (lo = i-P-1).

#include "tsx/window_mutants.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <deque>
#include <utility>

#include "core/selection.h"
#include "core/validity.h"
#include "ops/agg_internal.h"
#include "ops/join_internal.h"
#include "ops/sort.h"

namespace qe::mutant {

namespace detail = qe::ops::detail;

namespace {

std::vector<SortKey> sort_keys(const std::vector<std::uint32_t>& keys,
                               std::uint32_t time_col) {
    std::vector<SortKey> sk;
    sk.reserve(keys.size() + 1);
    for (std::uint32_t kc : keys)
        sk.push_back(SortKey{kc, SortDir::Asc, NullOrder::Last});
    sk.push_back(SortKey{time_col, SortDir::Asc, NullOrder::Last});
    return sk;
}

struct TimeVal {
    bool valid;
    std::int64_t t;
};
TimeVal read_time(const Column& c, std::uint32_t phys) {
    const bool valid = c.all_valid || validity::get_bit(c.validity, phys);
    if (!valid) return {false, 0};
    switch (c.type) {
        case Type::I32:
            return {true, reinterpret_cast<const std::int32_t*>(c.data)[phys]};
        case Type::I64:
        case Type::TS:
            return {true, reinterpret_cast<const std::int64_t*>(c.data)[phys]};
        default:
            return {false, 0};
    }
}

void put_time_word(OwnedColumn& c, Type t, std::size_t i, std::int64_t v) {
    std::byte* d = c.mutable_data();
    switch (t) {
        case Type::I32:
            reinterpret_cast<std::int32_t*>(d)[i] = static_cast<std::int32_t>(v);
            break;
        case Type::I64:
        case Type::TS:
            reinterpret_cast<std::int64_t*>(d)[i] = v;
            break;
        default:
            break;
    }
}

void fold(detail::AggCell& cell, AggFunc func, bool is_float,
          const detail::ColVal& v) {
    switch (func) {
        case AggFunc::CountStar: ++cell.cnt; return;
        case AggFunc::Count: if (v.valid) ++cell.cnt; return;
        case AggFunc::Sum:
        case AggFunc::Avg:
            if (!v.valid) return;
            if (is_float) cell.d += v.d; else cell.i += v.i;
            ++cell.cnt;
            return;
        case AggFunc::Min:
            if (!v.valid) return;
            if (is_float) cell.d = std::min(cell.d, v.d);
            else cell.i = std::min(cell.i, v.i);
            ++cell.cnt;
            return;
        case AggFunc::Max:
            if (!v.valid) return;
            if (is_float) cell.d = std::max(cell.d, v.d);
            else cell.i = std::max(cell.i, v.i);
            ++cell.cnt;
            return;
    }
}

detail::ColVal store_val(const detail::BuildStore& s, std::uint32_t col,
                         std::size_t row) {
    detail::ColVal r;
    r.valid = s.valid[col][row] != 0;
    if (!r.valid) return r;
    const std::byte* d = s.col_data(col);
    switch (s.types[col]) {
        case Type::I32:
            r.i = reinterpret_cast<const std::int32_t*>(d)[row];
            break;
        case Type::I64:
        case Type::TS:
            r.i = reinterpret_cast<const std::int64_t*>(d)[row];
            break;
        case Type::BOOL:
            r.i = reinterpret_cast<const std::uint8_t*>(d)[row];
            break;
        case Type::F64:
            r.d = reinterpret_cast<const double*>(d)[row];
            break;
        case Type::STR:  // WP-7b: STR out of window grammar (mutant path)
            break;
    }
    return r;
}

std::uint64_t canon_f64(double v) {
    if (v == 0.0) v = 0.0;
    if (std::isnan(v)) return 0x7ff8000000000000ull;
    std::uint64_t w;
    std::memcpy(&w, &v, 8);
    return w;
}

struct SlideAcc {
    std::int64_t sum_i = 0;
    std::int64_t cnt = 0;
    std::deque<std::size_t> dq_min;
    std::deque<std::size_t> dq_max;
    void reset() {
        sum_i = 0;
        cnt = 0;
        dq_min.clear();
        dq_max.clear();
    }
};

}  // namespace

struct Window::State {
    std::unique_ptr<HashTable> ht;
    std::vector<std::vector<detail::AggCell>> cells;
    std::size_t num_groups = 0;
    std::vector<Type> key_types;

    std::unique_ptr<Operator> sorted;
    detail::BuildStore store;
    std::vector<std::vector<detail::AggCell>> agg_cells;
    std::vector<std::uint32_t> gather_idx;
    std::vector<std::uint32_t> scratch_idx;

    std::vector<Type> input_types;
    std::vector<bool> is_float;
};

Window::Window(std::unique_ptr<Operator> child, qe::tsx::WindowMode mode,
               std::vector<std::uint32_t> keys, std::uint32_t time,
               std::int64_t param, std::vector<AggSpec> aggs, WindowMutation mut)
    : child_(std::move(child)),
      mode_(mode),
      keys_(std::move(keys)),
      time_(time),
      param_(param),
      aggs_(std::move(aggs)),
      mut_(mut) {
    child_schema_ = child_->output_schema();
}

Schema Window::output_schema() const {
    Schema s;
    if (mode_ == qe::tsx::WindowMode::Tumbling) {
        for (std::uint32_t kc : keys_) s.fields.push_back(child_schema_.fields[kc]);
        s.fields.emplace_back(child_schema_.fields[time_].first,
                              child_schema_.fields[time_].second);
    } else {
        for (const auto& f : child_schema_.fields) s.fields.push_back(f);
    }
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        s.fields.emplace_back(a.out_name, agg_result_type(a.func, in));
    }
    return s;
}

void Window::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    State& st = *state_;
    emit_cursor_ = 0;
    st.input_types.reserve(aggs_.size());
    st.is_float.reserve(aggs_.size());
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        st.input_types.push_back(in);
        st.is_float.push_back(in == Type::F64);
    }
    if (mode_ == qe::tsx::WindowMode::Tumbling) build_tumbling();
    else build_sliding();
}

void Window::build_tumbling() {
    State& st = *state_;
    const std::int64_t W = param_;
    const Type ttype = child_schema_.fields[time_].second;

    st.key_types.clear();
    for (std::uint32_t kc : keys_)
        st.key_types.push_back(child_schema_.fields[kc].second);
    st.key_types.push_back(ttype);
    st.ht = std::make_unique<HashTable>(st.key_types, NullPolicy::kEqual);
    st.cells.assign(aggs_.size(), {});

    child_->open();
    std::vector<Column> key_views;
    std::vector<std::uint32_t> gids;
    while (std::optional<Batch> in = child_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;

        const Column& tcol = b.cols[time_];
        OwnedColumn bucket = OwnedColumn::make(ttype, tcol.len);
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint32_t phys = sel_at(b.sel, k);
            const TimeVal tv = read_time(tcol, phys);
            if (!tv.valid) {
                bucket.set_null(phys);
                continue;
            }
            // DEFECT kBucketEdge: shift the bucket edge by one => rows near a
            // boundary fall in the wrong bucket (correct is `tv.t / W`).
            const std::int64_t bid = (mut_ == WindowMutation::kBucketEdge)
                                         ? (tv.t + 1) / W
                                         : tv.t / W;
            put_time_word(bucket, ttype, phys, bid * W);
        }
        const Column bucket_view = bucket.view();

        key_views.clear();
        key_views.reserve(keys_.size() + 1);
        for (std::uint32_t kc : keys_) key_views.push_back(b.cols[kc]);
        key_views.push_back(bucket_view);
        const KeyColumns kc{key_views.data(), key_views.size(), b.sel};

        gids.resize(n);
        st.ht->insert_or_find(kc, n, gids.data(), hash_path_);
        const std::size_t G = st.ht->num_groups();
        if (G > st.num_groups) {
            for (std::size_t a = 0; a < aggs_.size(); ++a) {
                st.cells[a].resize(G);
                for (std::size_t g = st.num_groups; g < G; ++g)
                    st.cells[a][g] =
                        detail::init_agg_cell(aggs_[a].func, st.is_float[a]);
            }
            st.num_groups = G;
        }

        for (std::size_t a = 0; a < aggs_.size(); ++a) {
            const AggSpec& spec = aggs_[a];
            const bool isf = st.is_float[a];
            auto& cells = st.cells[a];
            if (spec.func == AggFunc::CountStar) {
                for (std::size_t k = 0; k < n; ++k) ++cells[gids[k]].cnt;
                continue;
            }
            const Column& col = b.cols[spec.input_col];
            for (std::size_t k = 0; k < n; ++k)
                fold(cells[gids[k]], spec.func, isf,
                     detail::read_col(col, b.sel, k));
        }
    }
    child_->close();
}

std::optional<Batch> Window::next_tumbling() {
    State& st = *state_;
    const std::size_t G = st.num_groups;
    if (emit_cursor_ >= G) return std::nullopt;
    const std::size_t m = std::min(kOutBatch, G - emit_cursor_);
    const std::size_t nk = keys_.size();

    OwnedBatch out;
    for (std::size_t j = 0; j < nk + 1; ++j) {
        const Type kt = st.key_types[j];
        OwnedColumn c = OwnedColumn::make(kt, m);
        for (std::size_t r = 0; r < m; ++r) {
            const std::uint32_t g = static_cast<std::uint32_t>(emit_cursor_ + r);
            detail::write_key_cell(c, r, kt, st.ht->group_is_null(g, j),
                                   st.ht->group_key_word(g, j));
        }
        out.add_column(std::move(c));
    }
    for (std::size_t a = 0; a < aggs_.size(); ++a) {
        const Type rt = agg_result_type(aggs_[a].func, st.input_types[a]);
        OwnedColumn c = OwnedColumn::make(rt, m);
        for (std::size_t r = 0; r < m; ++r)
            detail::write_agg_cell(c, r, aggs_[a].func, st.is_float[a], rt,
                                   st.cells[a][emit_cursor_ + r]);
        out.add_column(std::move(c));
    }
    emit_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void Window::build_sliding() {
    State& st = *state_;
    const std::int64_t P = param_;

    st.sorted = std::make_unique<Sort>(std::move(child_), sort_keys(keys_, time_));
    std::vector<Type> ctypes;
    ctypes.reserve(child_schema_.fields.size());
    for (const auto& f : child_schema_.fields) ctypes.push_back(f.second);
    st.store.init(ctypes);

    st.sorted->open();
    while (std::optional<Batch> in = st.sorted->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        for (std::size_t k = 0; k < n; ++k)
            st.store.append_row(b, sel_at(b.sel, k));
    }
    st.sorted->close();

    const std::size_t nrows = st.store.nrows;
    const std::size_t na = aggs_.size();
    st.agg_cells.assign(na, std::vector<detail::AggCell>(nrows));
    if (nrows == 0) return;

    auto same_keys = [&](std::size_t r, std::size_t q) -> bool {
        for (std::uint32_t kc : keys_) {
            const detail::ColVal a = store_val(st.store, kc, r);
            const detail::ColVal c = store_val(st.store, kc, q);
            if (a.valid != c.valid) return false;
            if (!a.valid) continue;
            if (st.store.types[kc] == Type::F64) {
                if (canon_f64(a.d) != canon_f64(c.d)) return false;
            } else if (a.i != c.i) {
                return false;
            }
        }
        return true;
    };

    std::vector<SlideAcc> acc(na);
    auto less_val = [&](std::size_t a, std::size_t x, std::size_t y) -> bool {
        const std::uint32_t col = aggs_[a].input_col;
        const detail::ColVal vx = store_val(st.store, col, x);
        const detail::ColVal vy = store_val(st.store, col, y);
        return st.is_float[a] ? (vx.d < vy.d) : (vx.i < vy.i);
    };

    std::size_t pstart = 0;
    std::size_t lo = 0;
    for (std::size_t i = 0; i < nrows; ++i) {
        if (i > 0 && !same_keys(i, i - 1)) {
            pstart = i;
            lo = i;
            for (auto& a : acc) a.reset();
        }
        // DEFECT kFrameOffByOne: reach P+1 preceding rows (correct is P).
        const std::int64_t reach =
            (mut_ == WindowMutation::kFrameOffByOne) ? P + 1 : P;
        const std::int64_t back =
            std::min<std::int64_t>(reach, static_cast<std::int64_t>(i - pstart));
        const std::size_t lo_target = i - static_cast<std::size_t>(back);
        for (; lo < lo_target; ++lo) {
            for (std::size_t a = 0; a < na; ++a) {
                const AggSpec& spec = aggs_[a];
                if (spec.func == AggFunc::CountStar) continue;
                const detail::ColVal v = store_val(st.store, spec.input_col, lo);
                if (v.valid) {
                    acc[a].sum_i -= v.i;
                    --acc[a].cnt;
                }
                if (!acc[a].dq_min.empty() && acc[a].dq_min.front() == lo)
                    acc[a].dq_min.pop_front();
                if (!acc[a].dq_max.empty() && acc[a].dq_max.front() == lo)
                    acc[a].dq_max.pop_front();
            }
        }
        for (std::size_t a = 0; a < na; ++a) {
            const AggSpec& spec = aggs_[a];
            if (spec.func == AggFunc::CountStar) continue;
            const detail::ColVal v = store_val(st.store, spec.input_col, i);
            if (!v.valid) continue;
            acc[a].sum_i += v.i;
            ++acc[a].cnt;
            if (spec.func == AggFunc::Min) {
                while (!acc[a].dq_min.empty() && !less_val(a, acc[a].dq_min.back(), i))
                    acc[a].dq_min.pop_back();
                acc[a].dq_min.push_back(i);
            } else if (spec.func == AggFunc::Max) {
                while (!acc[a].dq_max.empty() && !less_val(a, i, acc[a].dq_max.back()))
                    acc[a].dq_max.pop_back();
                acc[a].dq_max.push_back(i);
            }
        }
        const std::int64_t frame_size =
            static_cast<std::int64_t>(i) - static_cast<std::int64_t>(lo) + 1;
        for (std::size_t a = 0; a < na; ++a) {
            const AggSpec& spec = aggs_[a];
            detail::AggCell cell = detail::init_agg_cell(spec.func, st.is_float[a]);
            switch (spec.func) {
                case AggFunc::CountStar:
                    cell.cnt = frame_size;
                    break;
                case AggFunc::Count:
                    cell.cnt = acc[a].cnt;
                    break;
                case AggFunc::Sum:
                case AggFunc::Avg:
                    cell.cnt = acc[a].cnt;
                    if (st.is_float[a]) {
                        double s = 0.0;
                        for (std::size_t r = lo; r <= i; ++r) {
                            const detail::ColVal v =
                                store_val(st.store, spec.input_col, r);
                            if (v.valid) s += v.d;
                        }
                        cell.d = s;
                    } else {
                        cell.i = acc[a].sum_i;
                    }
                    break;
                case AggFunc::Min:
                    cell.cnt = acc[a].cnt;
                    if (acc[a].cnt > 0) {
                        const detail::ColVal v = store_val(
                            st.store, spec.input_col, acc[a].dq_min.front());
                        if (st.is_float[a]) cell.d = v.d; else cell.i = v.i;
                    }
                    break;
                case AggFunc::Max:
                    cell.cnt = acc[a].cnt;
                    if (acc[a].cnt > 0) {
                        const detail::ColVal v = store_val(
                            st.store, spec.input_col, acc[a].dq_max.front());
                        if (st.is_float[a]) cell.d = v.d; else cell.i = v.i;
                    }
                    break;
            }
            st.agg_cells[a][i] = cell;
        }
    }
}

std::optional<Batch> Window::next_sliding() {
    State& st = *state_;
    const std::size_t nrows = st.store.nrows;
    if (emit_cursor_ >= nrows) return std::nullopt;
    const std::size_t m = std::min(kOutBatch, nrows - emit_cursor_);
    const std::size_t nchild = child_schema_.fields.size();

    st.gather_idx.resize(m);
    for (std::size_t r = 0; r < m; ++r)
        st.gather_idx[r] = static_cast<std::uint32_t>(emit_cursor_ + r);

    OwnedBatch out;
    for (std::size_t c = 0; c < nchild; ++c) {
        OwnedColumn oc = OwnedColumn::make(child_schema_.fields[c].second, m);
        detail::emit_build_column(oc, st.store, c, st.gather_idx.data(), m,
                                  gather_path_, st.scratch_idx);
        out.add_column(std::move(oc));
    }
    for (std::size_t a = 0; a < aggs_.size(); ++a) {
        const Type rt = agg_result_type(aggs_[a].func, st.input_types[a]);
        OwnedColumn oc = OwnedColumn::make(rt, m);
        for (std::size_t r = 0; r < m; ++r)
            detail::write_agg_cell(oc, r, aggs_[a].func, st.is_float[a], rt,
                                   st.agg_cells[a][emit_cursor_ + r]);
        out.add_column(std::move(oc));
    }
    emit_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

std::optional<Batch> Window::next() {
    assert(opened_ && "next() before open()");
    return mode_ == qe::tsx::WindowMode::Tumbling ? next_tumbling()
                                                  : next_sliding();
}

void Window::close() {
    state_.reset();
    opened_ = false;
    emit_cursor_ = 0;
}

}  // namespace qe::mutant
