//  WP-6: implementation of the TEST-ONLY mutant hash-join operator
//  (ops/join_mutants.h). A faithful copy of ops/join.cpp's build / pair-generation
//  / output loops, reusing the SAME ops/join_internal.h helpers (BuildStore + the
//  gather emitters), with exactly one planted defect per Mutation. The `// BUG:`
//  lines mark each deviation from the real operator.

#include "ops/join_mutants.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "core/selection.h"
#include "ops/hashtable.h"
#include "ops/join_internal.h"

namespace qe::mutant {

namespace detail = qe::ops::detail;

struct HashJoin::State {
    std::unique_ptr<HashTable> ht;
    detail::BuildStore store;
    std::vector<std::vector<std::uint32_t>> group_rows;
    std::vector<Type> build_key_types;
    std::vector<std::uint32_t> pair_probe;
    std::vector<std::uint32_t> pair_build;
    std::vector<std::uint32_t> gids;
    std::vector<Column> key_views;
    std::vector<std::uint32_t> scratch_idx;
};

HashJoin::HashJoin(std::unique_ptr<Operator> probe,
                   std::unique_ptr<Operator> build,
                   std::vector<std::uint32_t> probe_keys,
                   std::vector<std::uint32_t> build_keys, JoinType type,
                   Mutation mut)
    : probe_(std::move(probe)),
      build_(std::move(build)),
      probe_keys_(std::move(probe_keys)),
      build_keys_(std::move(build_keys)),
      type_(type),
      mut_(mut) {
    assert(probe_keys_.size() == build_keys_.size());
    assert(!probe_keys_.empty());
    probe_schema_ = probe_->output_schema();
    build_schema_ = build_->output_schema();
}

Schema HashJoin::output_schema() const {
    Schema s;
    for (const auto& f : probe_schema_.fields) s.fields.push_back(f);
    for (const auto& f : build_schema_.fields) s.fields.push_back(f);
    return s;
}

void HashJoin::build_side() {
    State& st = *state_;

    // The set of key columns used to build the table. The composite mutant uses
    // ONLY the first key column.
    std::vector<std::uint32_t> bkeys = build_keys_;
    if (mut_ == Mutation::kCompositeFirstKeyOnly && bkeys.size() > 1)
        bkeys.resize(1);  // BUG: ignore all but the first key column

    st.build_key_types.clear();
    for (std::uint32_t kc : bkeys)
        st.build_key_types.push_back(build_schema_.fields[kc].second);
    st.ht = std::make_unique<HashTable>(st.build_key_types,
                                        NullPolicy::kNeverMatch);

    std::vector<Type> build_types;
    for (const auto& f : build_schema_.fields) build_types.push_back(f.second);
    st.store.init(build_types);

    build_->open();
    std::vector<Column> key_views;
    std::vector<std::uint32_t> gids;
    while (std::optional<Batch> in = build_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;

        key_views.clear();
        for (std::uint32_t kc : bkeys) key_views.push_back(b.cols[kc]);
        const KeyColumns kc{key_views.data(), key_views.size(), b.sel};
        gids.resize(n);
        st.ht->insert_or_find(kc, n, gids.data());

        const std::size_t G = st.ht->num_groups();
        if (G > st.group_rows.size()) st.group_rows.resize(G);

        for (std::size_t k = 0; k < n; ++k) {
            const std::uint32_t p = sel_at(b.sel, k);
            const std::uint32_t row = st.store.append_row(b, p);
            st.group_rows[gids[k]].push_back(row);
        }
    }
    build_->close();
}

bool HashJoin::build_pairs_for_probe() {
    State& st = *state_;
    st.pair_probe.clear();
    st.pair_build.clear();

    cur_probe_ = probe_->next();
    if (!cur_probe_) return false;

    const Batch& b = *cur_probe_;
    const std::size_t n = b.row_count;

    std::vector<std::uint32_t> pkeys = probe_keys_;
    if (mut_ == Mutation::kCompositeFirstKeyOnly && pkeys.size() > 1)
        pkeys.resize(1);  // BUG: probe on the first key only

    st.key_views.clear();
    for (std::uint32_t kc : pkeys) st.key_views.push_back(b.cols[kc]);
    const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};

    st.gids.resize(n);
    st.ht->find(kc, n, st.gids.data());

    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t phys = sel_at(b.sel, k);
        const std::uint32_t g = st.gids[k];
        if (g != kNoGroup) {
            for (std::uint32_t br : st.group_rows[g]) {
                if (mut_ == Mutation::kDropProbeMatch && !dropped_one_) {
                    dropped_one_ = true;  // BUG: skip emitting one matched row
                    continue;
                }
                st.pair_probe.push_back(phys);
                st.pair_build.push_back(br);
            }
        } else if (type_ == JoinType::Left) {
            st.pair_probe.push_back(phys);
            if (mut_ == Mutation::kLeftWrongNull) {
                // BUG: emit build row 0's real values instead of NULLs.
                st.pair_build.push_back(0);
            } else {
                st.pair_build.push_back(detail::kNullBuildRow);
            }
        }
    }
    pair_cursor_ = 0;
    return true;
}

void HashJoin::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    probe_done_ = false;
    pair_cursor_ = 0;
    dropped_one_ = false;
    build_side();
    probe_->open();
}

std::optional<Batch> HashJoin::next() {
    assert(opened_);
    State& st = *state_;

    while (pair_cursor_ >= st.pair_probe.size()) {
        if (probe_done_) return std::nullopt;
        if (!build_pairs_for_probe()) {
            probe_done_ = true;
            return std::nullopt;
        }
    }

    const Batch& pb = *cur_probe_;
    const std::size_t total = st.pair_probe.size();
    const std::size_t avail = total - pair_cursor_;
    const std::size_t m = std::min<std::size_t>(kOutBatch, avail);
    std::size_t emit = m;
    if (mut_ == Mutation::kFanoutTailOffByOne && m == avail && m > 1) {
        // BUG: the FINAL chunk of this probe batch is one row short, and the
        // cursor still advances past it — so that row is dropped for good (not
        // deferred to the next call).
        emit = m - 1;
    }
    const std::uint32_t* probe_idx = st.pair_probe.data() + pair_cursor_;
    const std::uint32_t* build_idx = st.pair_build.data() + pair_cursor_;

    OwnedBatch out;
    for (std::size_t c = 0; c < probe_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(probe_schema_.fields[c].second, emit);
        detail::emit_probe_column(oc, pb.cols[c], probe_idx, emit,
                                  GatherPath::kVector);
        out.add_column(std::move(oc));
    }
    for (std::size_t c = 0; c < build_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(build_schema_.fields[c].second, emit);
        detail::emit_build_column(oc, st.store, c, build_idx, emit,
                                  GatherPath::kVector, st.scratch_idx);
        out.add_column(std::move(oc));
    }

    pair_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void HashJoin::close() {
    if (opened_) probe_->close();
    state_.reset();
    cur_probe_.reset();
    opened_ = false;
    probe_done_ = false;
    pair_cursor_ = 0;
}

}  // namespace qe::mutant
