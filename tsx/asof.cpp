//  WP-12: backward as-of join implementation. See tsx/asof.h.
//
//  BUILD (open()): wrap the build child in the frozen Sort (ops/sort.h) keyed by
//  (build keys..., build timestamp) ascending, drain it fully, and (1) insert each
//  row's key tuple into the frozen HashTable (ops/hashtable.h,
//  NullPolicy::kNeverMatch — a NULL key never matches) to get a stable group id,
//  (2) materialize ALL build columns into a dense BuildStore (ops/join_internal.h)
//  for the emit-time gather, and (3) append each MATCHABLE row (non-NULL key AND
//  non-NULL timestamp) to its group's row list. Because the stream is sorted by
//  timestamp, each group's list comes out TIMESTAMP-ASCENDING with no extra sort.
//
//  PROBE (next()): the probe child is likewise Sort-wrapped by (probe keys...,
//  probe timestamp), so a probe key's rows form one contiguous, timestamp-ascending
//  run. HashTable::find() maps each probe row to its build group id (or kNoGroup).
//  Within a group we MERGE with a per-key advancing cursor: walking probe rows in
//  ascending timestamp, a build cursor only moves forward to the last tb <= t (the
//  nearest preceding). The cursor + current group id persist across probe batches
//  (a group's run may straddle a batch boundary). Matches expand into
//  (probe_phys_row, build_store_row) pairs emitted in dense <= kOutBatch batches —
//  a LEFT-join unmatched probe row carries the kNullBuildRow sentinel.
//
//  GATHER: probe-row and build-row column materialization reuses the WP-1 selection
//  gather kernels (simd/gather_kernels.h) via the shared emitters in
//  ops/join_internal.h; the scalar==vector differential drives both paths
//  (GatherPath) and asserts identical output. The per-key MERGE itself is
//  sequential scalar control flow (the WP-6 probe-walk precedent — no vector twin).

#include "tsx/asof.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "core/selection.h"
#include "core/validity.h"
#include "ops/join_internal.h"  // BuildStore + emit_probe_column/emit_build_column
#include "ops/sort.h"

namespace qe::tsx {

namespace detail = qe::ops::detail;

namespace {

// Sort keys for one side: every partition key ascending, then the timestamp,
// ascending. NULLS LAST throughout (placement is irrelevant to correctness here —
// NULL keys/timestamps are skipped — but it keeps null-bearing rows out of the way
// at the tail of each group run). >=1 key always (the timestamp), so Sort's >=1
// precondition holds even with zero partition keys.
std::vector<SortKey> sort_keys(const std::vector<std::uint32_t>& keys,
                               std::uint32_t time_col) {
    std::vector<SortKey> sk;
    sk.reserve(keys.size() + 1);
    for (std::uint32_t kc : keys)
        sk.push_back(SortKey{kc, SortDir::Asc, NullOrder::Last});
    sk.push_back(SortKey{time_col, SortDir::Asc, NullOrder::Last});
    return sk;
}

// A timestamp value read as int64 (the comparison domain): I32 widened, I64/TS
// verbatim. (BOOL/F64 are not valid timestamp types — asserted.)
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
        case Type::STR:  // WP-7b: STR is out of the as-of grammar (no STR timestamps)
            assert(false && "asof timestamp column must be I32/I64/TS");
            return {false, 0};
    }
    return {false, 0};
}

// True iff logical row k (physical sel_at(sel,k)) has a NULL in ANY key column —
// such a row never matches (kNeverMatch), regardless of the hash group id.
bool any_key_null(const std::vector<Column>& key_views,
                  const SelectionVector* sel, std::size_t k) {
    const std::uint32_t phys = sel_at(sel, k);
    for (const Column& c : key_views)
        if (!c.all_valid && !validity::get_bit(c.validity, phys)) return true;
    return false;
}

}  // namespace

struct AsofJoin::State {
    std::unique_ptr<HashTable> ht;
    detail::BuildStore store;  // all build columns, in sorted (drained) order
    // Group id -> build-store row indices for that key, TIMESTAMP-ASCENDING (only
    // matchable rows: non-NULL key AND non-NULL timestamp).
    std::vector<std::vector<std::uint32_t>> group_rows;
    // Timestamp (int64) per build-store row; parallel to the store's rows.
    std::vector<std::int64_t> build_ts;
    std::vector<Type> build_key_types;

    // Pair arrays for the CURRENT probe batch (physical probe row, build-store row
    // | kNullBuildRow). Rebuilt per probe batch, drained in <= kOutBatch chunks.
    std::vector<std::uint32_t> pair_probe;
    std::vector<std::uint32_t> pair_build;

    // Per-key advancing-cursor merge state, persisted across probe batches.
    std::uint32_t last_gid = kNoGroup;  // group whose cursor `cursor` belongs to
    std::size_t cursor = 0;             // index into group_rows[last_gid]

    // Scratch reused across calls/chunks.
    std::vector<std::uint32_t> gids;        // probe find() output
    std::vector<Column> key_views;          // per-batch key column views
    std::vector<std::uint32_t> scratch_idx; // build gather safe-index scratch
};

AsofJoin::AsofJoin(std::unique_ptr<Operator> probe,
                   std::unique_ptr<Operator> build,
                   std::vector<std::uint32_t> probe_keys,
                   std::vector<std::uint32_t> build_keys,
                   std::uint32_t probe_time, std::uint32_t build_time,
                   AsofType type, std::optional<std::int64_t> tolerance)
    : probe_keys_(std::move(probe_keys)),
      build_keys_(std::move(build_keys)),
      probe_time_(probe_time),
      build_time_(build_time),
      type_(type),
      tolerance_(tolerance) {
    assert(probe_keys_.size() == build_keys_.size() &&
           "asof needs equal-length key lists");
    // Capture the children's schemas BEFORE wrapping (Sort preserves the schema, so
    // the wrapped operator reports the same columns/order — the asof output schema
    // and the key/time column indices are unchanged by the sort).
    probe_schema_ = probe->output_schema();
    build_schema_ = build->output_schema();
#ifndef NDEBUG
    for (std::size_t i = 0; i < probe_keys_.size(); ++i)
        assert(probe_schema_.fields[probe_keys_[i]].second ==
                   build_schema_.fields[build_keys_[i]].second &&
               "asof key column types must match positionally");
    assert(probe_time_ < probe_schema_.fields.size() &&
           build_time_ < build_schema_.fields.size() && "asof time col in range");
#endif
    // Reuse the frozen Sort to order each side by (keys..., timestamp). probe_ /
    // build_ become the SORTED children; the raw inputs are owned within them.
    probe_ = std::make_unique<Sort>(std::move(probe),
                                    sort_keys(probe_keys_, probe_time_));
    build_ = std::make_unique<Sort>(std::move(build),
                                    sort_keys(build_keys_, build_time_));
}

Schema AsofJoin::output_schema() const {
    Schema s;
    s.fields.reserve(probe_schema_.fields.size() + build_schema_.fields.size());
    for (const auto& f : probe_schema_.fields) s.fields.push_back(f);
    for (const auto& f : build_schema_.fields) s.fields.push_back(f);
    return s;
}

void AsofJoin::build_side() {
    State& st = *state_;

    // ZERO partition keys => one GLOBAL partition (group id 0); the frozen
    // HashTable requires >=1 key column, so we skip it entirely in that case.
    const bool keyed = !build_keys_.empty();
    if (keyed) {
        st.build_key_types.clear();
        for (std::uint32_t kc : build_keys_)
            st.build_key_types.push_back(build_schema_.fields[kc].second);
        st.ht = std::make_unique<HashTable>(st.build_key_types,
                                            NullPolicy::kNeverMatch);
    } else {
        st.group_rows.resize(1);  // the single global group
    }

    std::vector<Type> build_types;
    build_types.reserve(build_schema_.fields.size());
    for (const auto& f : build_schema_.fields) build_types.push_back(f.second);
    st.store.init(build_types);

    build_->open();  // Sort: drains + sorts the build child by (keys, ts)
    std::vector<Column> key_views;
    std::vector<std::uint32_t> gids;
    while (std::optional<Batch> in = build_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;

        if (keyed) {
            key_views.clear();
            key_views.reserve(build_keys_.size());
            for (std::uint32_t kc : build_keys_) key_views.push_back(b.cols[kc]);
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
            // A row is a match candidate only if its key is non-NULL (a live,
            // findable group) AND its timestamp is non-NULL. The sorted stream is
            // timestamp-ascending, so appends keep group_rows[gid] ascending.
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

    const bool keyed = !probe_keys_.empty();
    if (keyed) {
        st.key_views.clear();
        st.key_views.reserve(probe_keys_.size());
        for (std::uint32_t kc : probe_keys_) st.key_views.push_back(b.cols[kc]);
        const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};
        st.gids.resize(n);
        st.ht->find(kc, n, st.gids.data(), hash_path_);
    }

    const Column& tcol = b.cols[probe_time_];
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t phys = sel_at(b.sel, k);
        // Zero keys => the single global group (id 0); else the probed group id.
        const std::uint32_t gid = keyed ? st.gids[k] : 0u;

        std::uint32_t matched = detail::kNullBuildRow;
        const TimeVal tv = read_time(tcol, phys);
        // A probe row matches only with a non-NULL key, a non-NULL timestamp, and a
        // build group present for its key.
        if (gid != kNoGroup && tv.valid &&
            (!keyed || !any_key_null(st.key_views, b.sel, k))) {
            const std::vector<std::uint32_t>& blist = st.group_rows[gid];
            // New contiguous group run => reset the cursor to its start. (Sorted by
            // (key, ts), a key's probe rows are contiguous, so the cursor advances
            // monotonically within the run.)
            if (gid != st.last_gid) {
                st.last_gid = gid;
                st.cursor = 0;
            }
            // Advance to the first build ts STRICTLY GREATER than t; the last one
            // <= t (cursor-1) is the nearest preceding match. `<=` is the contract
            // boundary (equal timestamps match) — using `<` is the classic bug.
            while (st.cursor < blist.size() &&
                   st.build_ts[blist[st.cursor]] <= tv.t)
                ++st.cursor;
            if (st.cursor > 0) {
                const std::uint32_t cand = blist[st.cursor - 1];
                // Within-tolerance variant: keep the nearest preceding only if it
                // is no further back than the tolerance window.
                if (!tolerance_ || (tv.t - st.build_ts[cand]) <= *tolerance_)
                    matched = cand;
            }
        }

        if (matched != detail::kNullBuildRow) {
            st.pair_probe.push_back(phys);
            st.pair_build.push_back(matched);
        } else if (type_ == AsofType::Left) {
            st.pair_probe.push_back(phys);
            st.pair_build.push_back(detail::kNullBuildRow);
        }
        // INNER + no match: emit nothing.
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
    probe_->open();  // Sort: drains + sorts the probe child by (keys, ts)
}

std::optional<Batch> AsofJoin::next() {
    assert(opened_ && "next() before open()");
    State& st = *state_;

    // Advance to a probe batch that has at least one pair to emit (an inner-asof
    // batch can produce zero pairs — pull the next one).
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
    // Probe columns first (child order), then build columns (child order).
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
    // The build child (Sort) was closed at end-of-build. Close the probe child
    // (idempotent-friendly per the Operator contract), then release state.
    if (opened_) probe_->close();
    state_.reset();
    cur_probe_.reset();
    opened_ = false;
    probe_done_ = false;
    pair_cursor_ = 0;
}

}  // namespace qe::tsx
