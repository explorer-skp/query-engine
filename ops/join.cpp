//  WP-6: Hash equi-join implementation. See ops/join.h.
//
//  BUILD (open()): drain the build child fully, inserting its key columns into the
//  frozen HashTable (ops/hashtable.h) with NullPolicy::kNeverMatch (SQL join NULLs
//  never match). The table dedups distinct keys to ONE stable group id and stores
//  no payloads, so we build the multiplicity map group_id -> [build row indices]
//  ourselves and materialize the build columns into a dense BuildStore
//  (ops/join_internal.h) so they can be gathered at emit time.
//
//  PROBE (next()): pull probe batches; HashTable::find() (no insert) maps each
//  probe row to a group id or kNoGroup. We expand matches into (probe_phys_row,
//  build_row) pairs for the whole probe batch, then emit them in dense <=kOutBatch
//  output batches across successive next() calls — so a single high-fanout probe
//  row, or a LEFT-join unmatched row, is split at the row granularity and the
//  output-batch tail is always correct.
//
//  GATHER: the build-row and probe-row column materialization reuses the WP-1
//  selection-vector gather kernels (simd/gather_kernels.h) via the shared
//  emitters in ops/join_internal.h; the scalar==vector differential drives both
//  paths (GatherPath) and asserts identical output.

#include "ops/join.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "core/selection.h"
#include "ops/join_internal.h"

namespace qe {

namespace detail = qe::ops::detail;

struct HashJoin::State {
    std::unique_ptr<HashTable> ht;
    detail::BuildStore store;
    // Multiplicity map: group id -> list of build-store row indices for that key.
    std::vector<std::vector<std::uint32_t>> group_rows;

    std::vector<Type> build_key_types;

    // Pair arrays for the CURRENT probe batch: parallel vectors of (physical probe
    // row, build-store row | kNullBuildRow). Rebuilt per probe batch, drained in
    // <=kOutBatch chunks across next() calls.
    std::vector<std::uint32_t> pair_probe;
    std::vector<std::uint32_t> pair_build;

    // Scratch reused across emit chunks.
    std::vector<std::uint32_t> gids;        // probe find() output
    std::vector<Column> key_views;          // per-batch key column views
    std::vector<std::uint32_t> scratch_idx; // build gather safe-index scratch
};

HashJoin::HashJoin(std::unique_ptr<Operator> probe,
                   std::unique_ptr<Operator> build,
                   std::vector<std::uint32_t> probe_keys,
                   std::vector<std::uint32_t> build_keys, JoinType type)
    : probe_(std::move(probe)),
      build_(std::move(build)),
      probe_keys_(std::move(probe_keys)),
      build_keys_(std::move(build_keys)),
      type_(type) {
    assert(probe_keys_.size() == build_keys_.size() &&
           "join needs equal-length key lists");
    assert(!probe_keys_.empty() && "equi-join needs >=1 key");
    probe_schema_ = probe_->output_schema();
    build_schema_ = build_->output_schema();
#ifndef NDEBUG
    for (std::size_t i = 0; i < probe_keys_.size(); ++i)
        assert(probe_schema_.fields[probe_keys_[i]].second ==
                   build_schema_.fields[build_keys_[i]].second &&
               "join key column types must match positionally");
#endif
}

Schema HashJoin::output_schema() const {
    Schema s;
    s.fields.reserve(probe_schema_.fields.size() + build_schema_.fields.size());
    for (const auto& f : probe_schema_.fields) s.fields.push_back(f);
    for (const auto& f : build_schema_.fields) s.fields.push_back(f);
    return s;
}

void HashJoin::build_side() {
    State& st = *state_;

    // Key types from the BUILD side (the table is built/probed over these).
    st.build_key_types.clear();
    for (std::uint32_t kc : build_keys_)
        st.build_key_types.push_back(build_schema_.fields[kc].second);
    st.ht = std::make_unique<HashTable>(st.build_key_types,
                                        NullPolicy::kNeverMatch);

    // Materialize ALL build columns (probe carries no build payload otherwise).
    std::vector<Type> build_types;
    build_types.reserve(build_schema_.fields.size());
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
        key_views.reserve(build_keys_.size());
        for (std::uint32_t kc : build_keys_) key_views.push_back(b.cols[kc]);
        const KeyColumns kc{key_views.data(), key_views.size(), b.sel};

        gids.resize(n);
        st.ht->insert_or_find(kc, n, gids.data(), hash_path_);

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

    st.key_views.clear();
    st.key_views.reserve(probe_keys_.size());
    for (std::uint32_t kc : probe_keys_) st.key_views.push_back(b.cols[kc]);
    const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};

    st.gids.resize(n);
    st.ht->find(kc, n, st.gids.data(), hash_path_);

    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t phys = sel_at(b.sel, k);
        const std::uint32_t g = st.gids[k];
        if (g != kNoGroup) {
            // Matched: one output row per build row in this group (gather fan-out).
            for (std::uint32_t br : st.group_rows[g]) {
                st.pair_probe.push_back(phys);
                st.pair_build.push_back(br);
            }
        } else if (type_ == JoinType::Left) {
            // Unmatched probe row in a LEFT join: one row, build columns NULL.
            st.pair_probe.push_back(phys);
            st.pair_build.push_back(detail::kNullBuildRow);
        }
        // Inner + unmatched: emit nothing.
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

    build_side();
    probe_->open();
}

std::optional<Batch> HashJoin::next() {
    assert(opened_ && "next() before open()");
    State& st = *state_;

    // Advance to a probe batch that has at least one pair to emit. A probe batch
    // can produce zero pairs (inner join, all rows unmatched) — pull the next one.
    while (pair_cursor_ >= st.pair_probe.size()) {
        if (probe_done_) return std::nullopt;
        if (!build_pairs_for_probe()) {
            probe_done_ = true;
            return std::nullopt;
        }
    }

    const Batch& pb = *cur_probe_;
    const std::size_t total = st.pair_probe.size();
    const std::size_t m = std::min<std::size_t>(kOutBatch, total - pair_cursor_);
    const std::uint32_t* probe_idx = st.pair_probe.data() + pair_cursor_;
    const std::uint32_t* build_idx = st.pair_build.data() + pair_cursor_;

    OwnedBatch out;
    // Probe columns first (in child order), then build columns (in child order).
    for (std::size_t c = 0; c < probe_schema_.fields.size(); ++c) {
        OwnedColumn oc =
            OwnedColumn::make(probe_schema_.fields[c].second, m);
        detail::emit_probe_column(oc, pb.cols[c], probe_idx, m, gather_path_);
        out.add_column(std::move(oc));
    }
    for (std::size_t c = 0; c < build_schema_.fields.size(); ++c) {
        OwnedColumn oc =
            OwnedColumn::make(build_schema_.fields[c].second, m);
        detail::emit_build_column(oc, st.store, c, build_idx, m, gather_path_,
                                  st.scratch_idx);
        out.add_column(std::move(oc));
    }

    pair_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void HashJoin::close() {
    // Build child was closed at end-of-build. Close the probe child (idempotent-
    // friendly per the Operator contract whether or not it was fully drained),
    // then release state.
    if (opened_) probe_->close();
    state_.reset();
    cur_probe_.reset();
    opened_ = false;
    probe_done_ = false;
    pair_cursor_ = 0;
}

}  // namespace qe
