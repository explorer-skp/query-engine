//  WP-13: windowed / time-bucketed aggregation implementation. See tsx/window.h.
//
//  TUMBLING (build_tumbling/next_tumbling). open() drains the child fully. Per batch
//  it derives the integer bucket per row (bucket_start = (t / W) * W, typed as the
//  timestamp column's type) into a fresh column laid out to MATCH the batch's
//  physical layout, then GROUP BYs (partition keys…, bucket) through the frozen
//  HashTable (ops/hashtable.h, NullPolicy::kEqual — NULL partition keys group
//  together, SQL semantics) and accumulates per-group aggregate state with the
//  SHARED ops/agg_internal.h cell + reader + finalizer. The grouped scatter is
//  sequential scalar control flow (the WP-5 precedent — conflict-free SIMD scatter
//  is not portably expressible in Highway). next() emits one row per occupied
//  (keys…, bucket) in dense <= kOutBatch batches: the partition keys then the bucket
//  (read back from the table, exactly like ops/aggregate.h), then the aggregates.
//
//  SLIDING (build_sliding/next_sliding). open() SORTs the child by (keys…, t)
//  ascending through the frozen Sort (ops/sort.h) so each partition is one
//  contiguous, time-ascending run, drains the sorted stream into a dense BuildStore
//  (ops/join_internal.h), then sweeps each partition ONCE with an advancing cursor:
//  COUNT(col)/SUM(integer) are kept as a true RUNNING accumulator (exact add on
//  enter, subtract on leave); MIN/MAX use a monotonic index deque; SUM/AVG over a
//  FLOAT column are recomputed over the (bounded) frame slice in time order each row
//  (documented: incremental float subtraction would drift past the D11 tolerance
//  near cancellation, and recomputing in time order makes the engine and the
//  independent reference produce bit-identical float sums). The running value at
//  each row is finalized with the SHARED ops/agg_internal.h finalizer. next() emits
//  one output row per input row — ALL child columns GATHERED from the store through
//  the WP-1 selection gather kernels (simd/gather_kernels.h) via emit_build_column
//  (vector path + scalar twin, selected by GatherPath, so scalar==vector is an
//  end-to-end check), then the running-aggregate columns written directly.
//
//  REUSE, NOT REIMPLEMENT: the bucket-scatter and the ring-buffer cursor are scalar
//  by the WP-5/WP-6 precedent; the ONE vectorized kernel with a scalar twin is the
//  sliding output's child-column gather (GatherPath). Tumbling emits group keys read
//  back from the table (like ops/aggregate.h), so it has no gather — its scalar==
//  vector seam is the HashTable's vector-vs-scalar key hash (HashPath).

#include "tsx/window.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <deque>
#include <utility>

#include "core/selection.h"
#include "core/validity.h"
#include "ops/agg_internal.h"     // AggCell, ColVal, read_col, write_*_cell, init
#include "ops/join_internal.h"    // BuildStore + emit_build_column
#include "ops/sort.h"

namespace qe::tsx {

namespace detail = qe::ops::detail;

namespace {

// Sort keys for the sliding sort: every partition key ascending, then the timestamp
// ascending. NULLS LAST throughout (placement is irrelevant to correctness — equal
// keys are contiguous regardless — but groups null-bearing rows at the tail). >= 1
// key always (the timestamp), so Sort's >= 1 precondition holds even with zero
// partition keys.
std::vector<SortKey> sort_keys(const std::vector<std::uint32_t>& keys,
                               std::uint32_t time_col) {
    std::vector<SortKey> sk;
    sk.reserve(keys.size() + 1);
    for (std::uint32_t kc : keys)
        sk.push_back(SortKey{kc, SortDir::Asc, NullOrder::Last});
    sk.push_back(SortKey{time_col, SortDir::Asc, NullOrder::Last});
    return sk;
}

// A timestamp value read as int64 (the comparison/bucketing domain): I32 widened,
// I64/TS verbatim. (BOOL/F64 are not valid timestamp types — asserted.)
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
        case Type::F64:
        case Type::BOOL:
        case Type::STR:  // WP-7b: STR is out of the window grammar (no STR time)
            assert(false && "window timestamp column must be I32/I64/TS");
            return {false, 0};
    }
    return {false, 0};
}

// Write `v` (the bucket lower edge) into a column of timestamp type `t` at index i.
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
        case Type::F64:
        case Type::BOOL:
        case Type::STR:
            break;  // unreachable for a timestamp column
    }
}

// Fold one row's value into a per-group cell (the GROUPED scatter step, tumbling).
// Identical contract to ops/aggregate.cpp's scatter_row (SUM/MIN/MAX/AVG ignore
// NULLs; COUNT(*) counts the row regardless; COUNT(col) counts non-null).
void fold(detail::AggCell& cell, AggFunc func, bool is_float,
          const detail::ColVal& v) {
    switch (func) {
        case AggFunc::CountStar:
            ++cell.cnt;
            return;
        case AggFunc::Count:
            if (v.valid) ++cell.cnt;
            return;
        case AggFunc::Sum:
        case AggFunc::Avg:
            if (!v.valid) return;
            if (is_float) cell.d += v.d; else cell.i += v.i;
            ++cell.cnt;
            return;
        case AggFunc::Min:
            if (!v.valid) return;
            // NaN-greatest total order (ops/agg_internal.h, audit C2): matches
            // the Phase-1 Aggregate and DuckDB; raw std::min dropped NaNs and
            // the tumbling/sliding modes disagreed with each other.
            if (is_float) cell.d = detail::f64_min_total(cell.d, v.d);
            else cell.i = std::min(cell.i, v.i);
            ++cell.cnt;
            return;
        case AggFunc::Max:
            if (!v.valid) return;
            if (is_float) cell.d = detail::f64_max_total(cell.d, v.d);
            else cell.i = std::max(cell.i, v.i);
            ++cell.cnt;
            return;
    }
}

// Read one typed value from a BuildStore cell (col, row) (the sliding frame reader).
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
        case Type::STR:  // WP-7b: STR aggregation is out of the window grammar
            assert(false && "window value column must be numeric, not STR");
            break;
    }
    return r;
}

// Canonicalize an F64 key word so -0.0 == +0.0 and any NaN compares equal (matches
// the HashTable / reference key semantics; generators avoid these, but stay honest).
std::uint64_t canon_f64(double v) {
    if (v == 0.0) v = 0.0;
    if (std::isnan(v)) return 0x7ff8000000000000ull;
    std::uint64_t w;
    std::memcpy(&w, &v, 8);
    return w;
}

}  // namespace

struct Window::State {
    // ---- tumbling ----
    std::unique_ptr<HashTable> ht;
    std::vector<std::vector<detail::AggCell>> cells;  // [agg][group]
    std::size_t num_groups = 0;
    std::vector<Type> key_types;  // partition key types..., then the bucket type

    // ---- sliding ----
    std::unique_ptr<Operator> sorted;        // child wrapped in the frozen Sort
    detail::BuildStore store;                // all child columns, sorted order
    std::vector<std::vector<detail::AggCell>> agg_cells;  // [agg][row], finalized
    std::vector<std::uint32_t> gather_idx;   // emit scratch (build-store indices)
    std::vector<std::uint32_t> scratch_idx;  // emit_build_column safe-index scratch

    // ---- shared ----
    std::vector<Type> input_types;  // [agg]; I64 placeholder for CountStar
    std::vector<bool> is_float;     // [agg]
};

Window::Window(std::unique_ptr<Operator> child, WindowMode mode,
               std::vector<std::uint32_t> keys, std::uint32_t time,
               std::int64_t param, std::vector<AggSpec> aggs)
    : child_(std::move(child)),
      mode_(mode),
      keys_(std::move(keys)),
      time_(time),
      param_(param),
      aggs_(std::move(aggs)) {
    assert(!aggs_.empty() && "Window needs >=1 aggregate");
    child_schema_ = child_->output_schema();
    // Real checks in EVERY build (audit H5): the sliding path reads partition
    // keys and aggregate inputs through store_val, whose STR arm is a debug
    // assert — under NDEBUG a STR partition key read back 0 for every row, so
    // ALL partitions silently merged. STR is out of the window grammar; reject
    // it loudly at construction instead.
    for (std::uint32_t kc : keys_)
        if (child_schema_.fields[kc].second == Type::STR)
            throw std::invalid_argument(
                "Window: STR partition keys are not supported (out of grammar)");
    for (const auto& a : aggs_)
        if (a.func != AggFunc::CountStar &&
            child_schema_.fields[a.input_col].second == Type::STR)
            throw std::invalid_argument(
                "Window: STR aggregate inputs are not supported (out of grammar)");
    if (child_schema_.fields[time_].second == Type::F64 ||
        child_schema_.fields[time_].second == Type::BOOL ||
        child_schema_.fields[time_].second == Type::STR)
        throw std::invalid_argument(
            "Window: time column must be I32/I64/TS");
#ifndef NDEBUG
    for (std::uint32_t kc : keys_)
        assert(kc < child_schema_.fields.size() && "window key col in range");
    assert(time_ < child_schema_.fields.size() && "window time col in range");
    if (mode_ == WindowMode::Tumbling) assert(param_ > 0 && "tumbling width W > 0");
    else assert(param_ >= 0 && "sliding preceding P >= 0");
#endif
}

Schema Window::output_schema() const {
    Schema s;
    if (mode_ == WindowMode::Tumbling) {
        // [partition keys…, bucket_start, aggregate columns…]
        for (std::uint32_t kc : keys_) s.fields.push_back(child_schema_.fields[kc]);
        // The bucket carries the timestamp column's type; name it after that column.
        s.fields.emplace_back(child_schema_.fields[time_].first,
                              child_schema_.fields[time_].second);
    } else {
        // ALL child columns in order, then the running-aggregate columns.
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

    // Per-agg input type / float-ness (shared by both modes).
    st.input_types.reserve(aggs_.size());
    st.is_float.reserve(aggs_.size());
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        st.input_types.push_back(in);
        st.is_float.push_back(in == Type::F64);
    }

    if (mode_ == WindowMode::Tumbling) build_tumbling();
    else build_sliding();
}

// ---- TUMBLING ---------------------------------------------------------------

void Window::build_tumbling() {
    State& st = *state_;
    const std::int64_t W = param_;
    const Type ttype = child_schema_.fields[time_].second;

    // Key schema = partition key types..., then the bucket (timestamp type).
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

        // Derive the bucket lower edge per row into a column laid out to MATCH the
        // batch's physical layout (so it shares the batch's selection vector with the
        // partition-key views). Only physical slots referenced by `sel` are written
        // and later read by the table.
        const Column& tcol = b.cols[time_];
        OwnedColumn bucket = OwnedColumn::make(ttype, tcol.len);
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint32_t phys = sel_at(b.sel, k);
            const TimeVal tv = read_time(tcol, phys);
            if (!tv.valid) {
                bucket.set_null(phys);
                continue;
            }
            // bucket_start = (t / W) * W. For t >= 0, W > 0 this equals DuckDB's
            // t - (t % W) (divergence-free; negative t is out of the grammar).
            const std::int64_t bstart = (tv.t / W) * W;
            put_time_word(bucket, ttype, phys, bstart);
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
    if (emit_cursor_ >= G) return std::nullopt;  // empty input => zero rows
    const std::size_t m = std::min(kOutBatch, G - emit_cursor_);
    const std::size_t nk = keys_.size();

    OwnedBatch out;
    // Partition keys then the bucket — all read back from the table (key order is
    // partition keys..., bucket), exactly the ops/aggregate.h grouped emit.
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

// ---- SLIDING ----------------------------------------------------------------

namespace {

// Per-agg running-window accumulator for the sliding sweep. `sum_i`/`cnt` are the
// true running accumulator (integer SUM + non-null count); the deques are monotonic
// index runs for MIN/MAX. Float SUM/AVG are recomputed over the frame slice (see the
// file header), so `sum_i` is unused for them.
struct SlideAcc {
    std::int64_t sum_i = 0;
    std::int64_t cnt = 0;  // non-null values of the agg's input column in the frame
    std::deque<std::size_t> dq_min;  // increasing values front->back (min at front)
    std::deque<std::size_t> dq_max;  // decreasing values front->back (max at front)

    void reset() {
        sum_i = 0;
        cnt = 0;
        dq_min.clear();
        dq_max.clear();
    }
};

}  // namespace

void Window::build_sliding() {
    State& st = *state_;
    const std::int64_t P = param_;

    // SORT the child by (keys..., time) ascending; drain the sorted stream into a
    // dense store of ALL child columns (sorted order).
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

    // True iff sorted rows r and r-1 share the same partition key tuple (null-aware,
    // F64-canonical) — i.e. they are NOT a partition boundary.
    auto same_keys = [&](std::size_t r, std::size_t q) -> bool {
        for (std::uint32_t kc : keys_) {
            const detail::ColVal a = store_val(st.store, kc, r);
            const detail::ColVal c = store_val(st.store, kc, q);
            if (a.valid != c.valid) return false;
            if (!a.valid) continue;  // both null => equal in this column
            if (st.store.types[kc] == Type::F64) {
                if (canon_f64(a.d) != canon_f64(c.d)) return false;
            } else if (a.i != c.i) {
                return false;
            }
        }
        return true;
    };

    std::vector<SlideAcc> acc(na);
    // Per agg, value comparison for the MIN/MAX deques (by the input column's type).
    auto less_val = [&](std::size_t a, std::size_t x, std::size_t y) -> bool {
        const std::uint32_t col = aggs_[a].input_col;
        const detail::ColVal vx = store_val(st.store, col, x);
        const detail::ColVal vy = store_val(st.store, col, y);
        // F64 compares under the NaN-greatest TOTAL order (audit C2). A raw `<`
        // made the monotonic deques order-dependent for NaN: frame {3.0, NaN}
        // reported NaN for MIN while {NaN, 3.0} reported 3.0. Under the total
        // order both report 3.0 (and MAX reports NaN), matching DuckDB.
        return st.is_float[a] ? detail::f64_less_total(vx.d, vy.d)
                              : (vx.i < vy.i);
    };

    std::size_t pstart = 0;
    std::size_t lo = 0;
    for (std::size_t i = 0; i < nrows; ++i) {
        if (i > 0 && !same_keys(i, i - 1)) {  // new partition
            pstart = i;
            lo = i;
            for (auto& a : acc) a.reset();
        }
        // Advance lo to max(pstart, i - P): rows leaving the frame.
        const std::int64_t back =
            std::min<std::int64_t>(P, static_cast<std::int64_t>(i - pstart));
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
        // Add row i.
        for (std::size_t a = 0; a < na; ++a) {
            const AggSpec& spec = aggs_[a];
            if (spec.func == AggFunc::CountStar) continue;
            const detail::ColVal v = store_val(st.store, spec.input_col, i);
            if (!v.valid) continue;
            acc[a].sum_i += v.i;
            ++acc[a].cnt;
            if (spec.func == AggFunc::Min) {
                while (!acc[a].dq_min.empty() && !less_val(a, acc[a].dq_min.back(), i))
                    acc[a].dq_min.pop_back();  // pop back values >= v (equal: keep NEWEST)
                acc[a].dq_min.push_back(i);
            } else if (spec.func == AggFunc::Max) {
                while (!acc[a].dq_max.empty() && !less_val(a, i, acc[a].dq_max.back()))
                    acc[a].dq_max.pop_back();  // pop back values <= v (equal: keep NEWEST)
                acc[a].dq_max.push_back(i);
            }
        }
        // Finalize each agg over the frame [lo, i].
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
                        double s = 0.0;  // fresh float sum over the frame, in order
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

    // The output rows ARE store rows [emit_cursor_, emit_cursor_+m) in order.
    st.gather_idx.resize(m);
    for (std::size_t r = 0; r < m; ++r)
        st.gather_idx[r] = static_cast<std::uint32_t>(emit_cursor_ + r);

    OwnedBatch out;
    // Child columns: GATHER from the store (vector path + scalar twin via GatherPath).
    for (std::size_t c = 0; c < nchild; ++c) {
        OwnedColumn oc = OwnedColumn::make(child_schema_.fields[c].second, m);
        detail::emit_build_column(oc, st.store, c, st.gather_idx.data(), m,
                                  gather_path_, st.scratch_idx);
        out.add_column(std::move(oc));
    }
    // Running-aggregate columns: written directly from the finalized cells.
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

// ---- dispatch ---------------------------------------------------------------

std::optional<Batch> Window::next() {
    assert(opened_ && "next() before open()");
    return mode_ == WindowMode::Tumbling ? next_tumbling() : next_sliding();
}

void Window::close() {
    state_.reset();
    opened_ = false;
    emit_cursor_ = 0;
}

}  // namespace qe::tsx
