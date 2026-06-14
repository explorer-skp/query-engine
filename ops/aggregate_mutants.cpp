//  WP-5: implementation of the TEST-ONLY mutant aggregation operator
//  (ops/aggregate_mutants.h). A faithful copy of ops/aggregate.cpp's GROUPED
//  drain / scatter / finalize / output loops, reusing the SAME ops/agg_internal.h
//  helpers, with exactly one planted defect per Mutation. The `// BUG:` lines
//  mark each deviation from the real operator.

#include "ops/aggregate_mutants.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "ops/agg_internal.h"
#include "ops/hashtable.h"

namespace qe::mutant {

namespace detail = qe::ops::detail;

struct Aggregate::State {
    std::unique_ptr<HashTable> ht;
    std::vector<std::vector<detail::AggCell>> cells;  // [agg][group]
    std::size_t num_groups = 0;
    std::vector<Type> key_types;
    std::vector<Type> input_types;
    std::vector<bool> is_float;
};

Aggregate::Aggregate(std::unique_ptr<Operator> child,
                     std::vector<std::uint32_t> key_cols,
                     std::vector<AggSpec> aggs, Mutation mut)
    : child_(std::move(child)),
      key_cols_(std::move(key_cols)),
      aggs_(std::move(aggs)),
      mut_(mut) {
    assert(!key_cols_.empty() && "mutant aggregate is grouped-only");
    assert(!aggs_.empty());
    child_schema_ = child_->output_schema();
}

Schema Aggregate::output_schema() const {
    Schema s;
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

void Aggregate::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    State& st = *state_;
    for (const auto& a : aggs_) {
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : child_schema_.fields[a.input_col].second;
        st.input_types.push_back(in);
        st.is_float.push_back(in == Type::F64);
    }
    for (std::uint32_t kc : key_cols_)
        st.key_types.push_back(child_schema_.fields[kc].second);
    st.ht = std::make_unique<HashTable>(st.key_types, NullPolicy::kEqual);
    st.cells.resize(aggs_.size());

    drain_and_build();
    emit_cursor_ = 0;
}

void Aggregate::drain_and_build() {
    State& st = *state_;
    child_->open();
    std::vector<Column> key_views;
    std::vector<std::uint32_t> gids;

    while (std::optional<Batch> in = child_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;

        key_views.clear();
        for (std::uint32_t kc : key_cols_) key_views.push_back(b.cols[kc]);
        const KeyColumns kc{key_views.data(), key_views.size(), b.sel};
        gids.resize(n);
        st.ht->insert_or_find(kc, n, gids.data());

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
            for (std::size_t k = 0; k < n; ++k) {
                const detail::ColVal v = detail::read_col(col, b.sel, k);
                detail::AggCell& cell = cells[gids[k]];
                switch (spec.func) {
                    case AggFunc::Count:
                        if (v.valid) ++cell.cnt;
                        break;
                    case AggFunc::Sum:
                    case AggFunc::Avg:
                        if (mut_ == Mutation::kFoldNullInSum) {
                            // BUG: fold a NULL as 0 AND count it, instead of
                            // ignoring it. All-null group => emits 0/seen, not
                            // NULL; COUNT/AVG denominators wrong.
                            if (isf)
                                cell.d += (v.valid ? v.d : 0.0);
                            else
                                cell.i += (v.valid ? v.i : 0);
                            ++cell.cnt;
                        } else {
                            if (!v.valid) break;
                            if (isf)
                                cell.d += v.d;
                            else
                                cell.i += v.i;
                            ++cell.cnt;
                        }
                        break;
                    case AggFunc::Min:
                        if (!v.valid) break;
                        if (isf)
                            cell.d = std::min(cell.d, v.d);
                        else
                            cell.i = std::min(cell.i, v.i);
                        ++cell.cnt;
                        break;
                    case AggFunc::Max:
                        if (!v.valid) break;
                        if (isf)
                            cell.d = std::max(cell.d, v.d);
                        else
                            cell.i = std::max(cell.i, v.i);
                        ++cell.cnt;
                        break;
                    case AggFunc::CountStar:
                        break;  // handled above
                }
            }
        }
    }
    child_->close();
}

namespace {
// Finalize with the kEmptyGroupZero defect: when set, emit the accumulator even
// for a zero-non-null group instead of NULL.
void write_agg_cell_mut(OwnedColumn& out, std::size_t r, AggFunc func,
                        bool is_float, Type rt, const detail::AggCell& c,
                        bool empty_group_zero) {
    if (empty_group_zero && c.cnt == 0 &&
        (func == AggFunc::Sum || func == AggFunc::Min || func == AggFunc::Max ||
         func == AggFunc::Avg)) {
        // BUG: emit a concrete value (0 / identity / 0.0) instead of NULL.
        std::byte* d = out.mutable_data();
        if (func == AggFunc::Avg) {
            reinterpret_cast<double*>(d)[r] = 0.0;
            return;
        }
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
        }
        return;
    }
    detail::write_agg_cell(out, r, func, is_float, rt, c);
}
}  // namespace

std::optional<Batch> Aggregate::next() {
    assert(opened_);
    State& st = *state_;

    std::size_t G = st.num_groups;
    if (mut_ == Mutation::kGroupTailOffByOne && G > 0) {
        // BUG: drop the last group (off-by-one on the group range), so the final
        // output batch is short one row.
        G = G - 1;
    }
    if (emit_cursor_ >= G) return std::nullopt;
    const std::size_t m = std::min(kOutBatch, G - emit_cursor_);

    OwnedBatch out;
    for (std::size_t j = 0; j < key_cols_.size(); ++j) {
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
            write_agg_cell_mut(c, r, aggs_[a].func, st.is_float[a], rt,
                               st.cells[a][emit_cursor_ + r],
                               mut_ == Mutation::kEmptyGroupZero);
        out.add_column(std::move(c));
    }
    emit_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void Aggregate::close() {
    state_.reset();
    opened_ = false;
}

}  // namespace qe::mutant
