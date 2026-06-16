//  WP-12 mutation self-test support. See tsx/asof_mutants.h. A faithful copy of
//  tsx/asof.cpp's build / merge / emit loops reusing the SAME ops/join_internal.h
//  gather emitters + frozen Sort/HashTable, differing by exactly one planted defect.

#include "tsx/asof_mutants.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "core/selection.h"
#include "core/validity.h"
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

bool any_key_null(const std::vector<Column>& kv, const SelectionVector* sel,
                  std::size_t k) {
    const std::uint32_t phys = sel_at(sel, k);
    for (const Column& c : kv)
        if (!c.all_valid && !validity::get_bit(c.validity, phys)) return true;
    return false;
}

// Effective key list under the kIgnoreLastKey defect: drop the last key column.
std::vector<std::uint32_t> effective_keys(const std::vector<std::uint32_t>& keys,
                                          AsofMutation mut) {
    if (mut == AsofMutation::kIgnoreLastKey && keys.size() > 1)
        return std::vector<std::uint32_t>(keys.begin(), keys.end() - 1);
    return keys;
}

}  // namespace

struct AsofJoin::State {
    std::unique_ptr<HashTable> ht;
    detail::BuildStore store;
    std::vector<std::vector<std::uint32_t>> group_rows;
    std::vector<std::int64_t> build_ts;
    std::vector<Type> build_key_types;
    std::vector<std::uint32_t> eff_probe_keys;
    std::vector<std::uint32_t> eff_build_keys;
    std::vector<std::uint32_t> pair_probe;
    std::vector<std::uint32_t> pair_build;
    std::uint32_t last_gid = kNoGroup;
    std::size_t cursor = 0;
    std::vector<std::uint32_t> gids;
    std::vector<Column> key_views;
    std::vector<std::uint32_t> scratch_idx;
};

AsofJoin::AsofJoin(std::unique_ptr<Operator> probe,
                   std::unique_ptr<Operator> build,
                   std::vector<std::uint32_t> probe_keys,
                   std::vector<std::uint32_t> build_keys,
                   std::uint32_t probe_time, std::uint32_t build_time,
                   qe::tsx::AsofType type,
                   std::optional<std::int64_t> tolerance, AsofMutation mut)
    : probe_keys_(std::move(probe_keys)),
      build_keys_(std::move(build_keys)),
      probe_time_(probe_time),
      build_time_(build_time),
      type_(type),
      tolerance_(tolerance),
      mut_(mut) {
    probe_schema_ = probe->output_schema();
    build_schema_ = build->output_schema();
    // The Sort uses the EFFECTIVE keys so the operator stays internally consistent
    // (under kIgnoreLastKey it partitions/orders by the reduced key set — the whole
    // point of the defect).
    probe_ = std::make_unique<Sort>(
        std::move(probe),
        sort_keys(effective_keys(probe_keys_, mut_), probe_time_));
    build_ = std::make_unique<Sort>(
        std::move(build),
        sort_keys(effective_keys(build_keys_, mut_), build_time_));
}

Schema AsofJoin::output_schema() const {
    Schema s;
    for (const auto& f : probe_schema_.fields) s.fields.push_back(f);
    for (const auto& f : build_schema_.fields) s.fields.push_back(f);
    return s;
}

void AsofJoin::build_side() {
    State& st = *state_;
    st.eff_build_keys = effective_keys(build_keys_, mut_);
    st.eff_probe_keys = effective_keys(probe_keys_, mut_);
    const bool keyed = !st.eff_build_keys.empty();

    if (keyed) {
        st.build_key_types.clear();
        for (std::uint32_t kc : st.eff_build_keys)
            st.build_key_types.push_back(build_schema_.fields[kc].second);
        st.ht = std::make_unique<HashTable>(st.build_key_types,
                                            NullPolicy::kNeverMatch);
    } else {
        st.group_rows.resize(1);
    }

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
        if (keyed) {
            key_views.clear();
            for (std::uint32_t kc : st.eff_build_keys) key_views.push_back(b.cols[kc]);
            const KeyColumns kc{key_views.data(), key_views.size(), b.sel};
            gids.resize(n);
            st.ht->insert_or_find(kc, n, gids.data(), hash_path_);
            const std::size_t G = st.ht->num_groups();
            if (G > st.group_rows.size()) st.group_rows.resize(G);
        }
        const Column& tcol = b.cols[build_time_];
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint32_t phys = sel_at(b.sel, k);
            const std::uint32_t row = st.store.append_row(b, phys);
            const TimeVal tv = read_time(tcol, phys);
            st.build_ts.push_back(tv.valid ? tv.t : 0);
            const std::uint32_t gid = keyed ? gids[k] : 0u;
            if ((!keyed || !any_key_null(key_views, b.sel, k)) && tv.valid)
                st.group_rows[gid].push_back(row);
        }
    }
    build_->close();
}

bool AsofJoin::build_pairs_for_probe() {
    State& st = *state_;
    st.pair_probe.clear();
    st.pair_build.clear();
    cur_probe_ = probe_->next();
    if (!cur_probe_) return false;
    const Batch& b = *cur_probe_;
    const std::size_t n = b.row_count;
    const bool keyed = !st.eff_probe_keys.empty();
    if (keyed) {
        st.key_views.clear();
        for (std::uint32_t kc : st.eff_probe_keys) st.key_views.push_back(b.cols[kc]);
        const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};
        st.gids.resize(n);
        st.ht->find(kc, n, st.gids.data(), hash_path_);
    }

    const Column& tcol = b.cols[probe_time_];
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t phys = sel_at(b.sel, k);
        const std::uint32_t gid = keyed ? st.gids[k] : 0u;
        std::uint32_t matched = detail::kNullBuildRow;
        const TimeVal tv = read_time(tcol, phys);
        if (gid != kNoGroup && tv.valid &&
            (!keyed || !any_key_null(st.key_views, b.sel, k))) {
            const std::vector<std::uint32_t>& blist = st.group_rows[gid];
            if (gid != st.last_gid) {
                st.last_gid = gid;
                st.cursor = 0;
            }
            // DEFECT kBoundaryStrict: `<` drops equal-timestamp matches.
            if (mut_ == AsofMutation::kBoundaryStrict) {
                while (st.cursor < blist.size() &&
                       st.build_ts[blist[st.cursor]] < tv.t)
                    ++st.cursor;
            } else {
                while (st.cursor < blist.size() &&
                       st.build_ts[blist[st.cursor]] <= tv.t)
                    ++st.cursor;
            }
            if (mut_ == AsofMutation::kNearestFollowing) {
                // DEFECT: nearest FOLLOWING (cursor) instead of preceding (cursor-1).
                if (st.cursor < blist.size()) {
                    const std::uint32_t cand = blist[st.cursor];
                    if (!tolerance_ ||
                        (st.build_ts[cand] - tv.t) <= *tolerance_)
                        matched = cand;
                }
            } else if (st.cursor > 0) {
                const std::uint32_t cand = blist[st.cursor - 1];
                if (!tolerance_ || (tv.t - st.build_ts[cand]) <= *tolerance_)
                    matched = cand;
            }
        }

        if (matched != detail::kNullBuildRow) {
            st.pair_probe.push_back(phys);
            st.pair_build.push_back(matched);
        } else if (type_ == qe::tsx::AsofType::Left) {
            st.pair_probe.push_back(phys);
            // DEFECT kLeftWrongNull: emit build row 0's real values, not NULL.
            st.pair_build.push_back(
                (mut_ == AsofMutation::kLeftWrongNull && st.store.nrows > 0)
                    ? 0u
                    : detail::kNullBuildRow);
        }
    }
    pair_cursor_ = 0;
    return true;
}

void AsofJoin::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    probe_done_ = false;
    pair_cursor_ = 0;
    build_side();
    probe_->open();
}

std::optional<Batch> AsofJoin::next() {
    assert(opened_ && "next() before open()");
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
    const std::size_t m = std::min<std::size_t>(kOutBatch, total - pair_cursor_);
    const std::uint32_t* probe_idx = st.pair_probe.data() + pair_cursor_;
    const std::uint32_t* build_idx = st.pair_build.data() + pair_cursor_;

    OwnedBatch out;
    for (std::size_t c = 0; c < probe_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(probe_schema_.fields[c].second, m);
        detail::emit_probe_column(oc, pb.cols[c], probe_idx, m, gather_path_);
        out.add_column(std::move(oc));
    }
    for (std::size_t c = 0; c < build_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(build_schema_.fields[c].second, m);
        detail::emit_build_column(oc, st.store, c, build_idx, m, gather_path_,
                                  st.scratch_idx);
        out.add_column(std::move(oc));
    }
    pair_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void AsofJoin::close() {
    if (opened_) probe_->close();
    state_.reset();
    cur_probe_.reset();
    opened_ = false;
    probe_done_ = false;
    pair_cursor_ = 0;
}

}  // namespace qe::mutant
