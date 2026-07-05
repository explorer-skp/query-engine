//  WP-7b mutation library implementation. See ops/string_key_mutants.h. A faithful
//  copy of ops/join.cpp whose ONLY variable is how STR keys reach the hash table:
//   * kNone        — canonicalize STR keys to a shared VALUE-id space (real path).
//   * kHashRawCode — feed the RAW int32 codes (the planted "by code" defect): two
//     equal strings with different per-side dictionary codes miss the join.
//  Everything else (BuildStore, multiplicity map, gather/emit) is the shared,
//  unmutated join plumbing (ops/join_internal.h), so the differential isolates the
//  key-wiring defect.

#include "ops/string_key_mutants.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>

#include "core/selection.h"
#include "core/string_dict.h"
#include "core/validity.h"
#include "ops/join_internal.h"

namespace qe::mutant {

namespace detail = qe::ops::detail;

namespace {
// Canonicalize a STR key column to a dense-by-physical-layout I32 value-id column
// (real path). Identical to ops/join.cpp's helper.
OwnedColumn canonicalize_str_key_to_i32(const Column& src,
                                        const SelectionVector* sel,
                                        StringDict& idmap) {
    const std::size_t len = src.len;
    OwnedColumn out = OwnedColumn::make(Type::I32, len);
    auto* ids = reinterpret_cast<std::int32_t*>(out.mutable_data());
    const auto* codes = reinterpret_cast<const std::int32_t*>(src.data);
    // Audit H3: touch ONLY the batch's live rows — unselected physical slots
    // need not hold meaningful codes (Batch contract), so resolving them is UB
    // and interning them pollutes the shared value-id dict. Physical layout is
    // preserved (the batch's sel keeps indexing the output); dead slots get id
    // 0 and are never read by any sel-aware consumer.
    std::memset(ids, 0, len * sizeof(std::int32_t));
    const std::size_t n = sel ? sel->len : len;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = sel_at(sel, k);
        const bool valid = src.all_valid || validity::get_bit(src.validity, p);
        if (valid)
            ids[p] = idmap.intern(src.dict->at(codes[p]));
        else
            out.set_null(p);
    }
    return out;
}
}  // namespace

struct StringKeyJoin::State {
    std::unique_ptr<HashTable> ht;
    detail::BuildStore store;
    std::vector<std::vector<std::uint32_t>> group_rows;
    std::vector<Type> build_key_types;
    std::vector<std::shared_ptr<StringDict>> key_id_maps;  // kNone only
    std::vector<OwnedColumn> canon_keys;

    std::vector<std::uint32_t> pair_probe;
    std::vector<std::uint32_t> pair_build;
    std::vector<std::uint32_t> gids;
    std::vector<Column> key_views;
    std::vector<std::uint32_t> scratch_idx;
};

StringKeyJoin::StringKeyJoin(std::unique_ptr<Operator> probe,
                             std::unique_ptr<Operator> build,
                             std::vector<std::uint32_t> probe_keys,
                             std::vector<std::uint32_t> build_keys,
                             qe::JoinType type, StringKeyMutation mutation)
    : probe_(std::move(probe)),
      build_(std::move(build)),
      probe_keys_(std::move(probe_keys)),
      build_keys_(std::move(build_keys)),
      type_(type),
      mutation_(mutation) {
    assert(probe_keys_.size() == build_keys_.size());
    assert(!probe_keys_.empty());
    probe_schema_ = probe_->output_schema();
    build_schema_ = build_->output_schema();
}

Schema StringKeyJoin::output_schema() const {
    Schema s;
    s.fields.reserve(probe_schema_.fields.size() + build_schema_.fields.size());
    for (const auto& f : probe_schema_.fields) s.fields.push_back(f);
    for (const auto& f : build_schema_.fields) s.fields.push_back(f);
    return s;
}

// Assemble the key views for a batch, applying (or skipping) STR canonicalization.
void StringKeyJoin::make_key_views(const std::vector<std::uint32_t>& keys,
                                   const Batch& b) {
    State& st = *state_;
    st.key_views.clear();
    st.key_views.reserve(keys.size());
    st.canon_keys.clear();
    st.canon_keys.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const Column& col = b.cols[keys[i]];
        const bool is_str = col.type == Type::STR;
        if (is_str && mutation_ == StringKeyMutation::kNone) {
            st.canon_keys.push_back(
                canonicalize_str_key_to_i32(col, b.sel, *st.key_id_maps[i]));
            st.key_views.push_back(st.canon_keys.back().view());
        } else {
            // kHashRawCode (the bug): pass the raw STR column — the table hashes
            // raw codes, so cross-dictionary equal strings miss.
            st.key_views.push_back(col);
        }
    }
}

void StringKeyJoin::build_side() {
    State& st = *state_;
    st.build_key_types.clear();
    st.key_id_maps.assign(build_keys_.size(), nullptr);
    for (std::size_t i = 0; i < build_keys_.size(); ++i) {
        const Type t = build_schema_.fields[build_keys_[i]].second;
        if (t == Type::STR && mutation_ == StringKeyMutation::kNone) {
            st.key_id_maps[i] = std::make_shared<StringDict>();
            st.build_key_types.push_back(Type::I32);
        } else {
            st.build_key_types.push_back(t);  // raw (STR stays STR for the mutant)
        }
    }
    st.ht = std::make_unique<HashTable>(st.build_key_types,
                                        NullPolicy::kNeverMatch);

    std::vector<Type> build_types;
    for (const auto& f : build_schema_.fields) build_types.push_back(f.second);
    st.store.init(build_types);

    build_->open();
    std::vector<std::uint32_t> gids;
    while (std::optional<Batch> in = build_->next()) {
        const Batch& b = *in;
        const std::size_t n = b.row_count;
        if (n == 0) continue;
        make_key_views(build_keys_, b);
        const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};
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

bool StringKeyJoin::build_pairs_for_probe() {
    State& st = *state_;
    st.pair_probe.clear();
    st.pair_build.clear();
    cur_probe_ = probe_->next();
    if (!cur_probe_) return false;
    const Batch& b = *cur_probe_;
    const std::size_t n = b.row_count;
    make_key_views(probe_keys_, b);
    const KeyColumns kc{st.key_views.data(), st.key_views.size(), b.sel};
    st.gids.resize(n);
    st.ht->find(kc, n, st.gids.data());
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t phys = sel_at(b.sel, k);
        const std::uint32_t g = st.gids[k];
        if (g != kNoGroup) {
            for (std::uint32_t br : st.group_rows[g]) {
                st.pair_probe.push_back(phys);
                st.pair_build.push_back(br);
            }
        } else if (type_ == qe::JoinType::Left) {
            st.pair_probe.push_back(phys);
            st.pair_build.push_back(detail::kNullBuildRow);
        }
    }
    pair_cursor_ = 0;
    return true;
}

void StringKeyJoin::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    probe_done_ = false;
    pair_cursor_ = 0;
    build_side();
    probe_->open();
}

std::optional<Batch> StringKeyJoin::next() {
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
    const std::size_t m = std::min<std::size_t>(kOutBatch, total - pair_cursor_);
    const std::uint32_t* probe_idx = st.pair_probe.data() + pair_cursor_;
    const std::uint32_t* build_idx = st.pair_build.data() + pair_cursor_;

    OwnedBatch out;
    for (std::size_t c = 0; c < probe_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(probe_schema_.fields[c].second, m);
        detail::emit_probe_column(oc, pb.cols[c], probe_idx, m,
                                  qe::GatherPath::kVector);
        out.add_column(std::move(oc));
    }
    for (std::size_t c = 0; c < build_schema_.fields.size(); ++c) {
        OwnedColumn oc = OwnedColumn::make(build_schema_.fields[c].second, m);
        detail::emit_build_column(oc, st.store, c, build_idx, m,
                                  qe::GatherPath::kVector, st.scratch_idx);
        out.add_column(std::move(oc));
    }
    pair_cursor_ += m;
    current_ = std::move(out);
    return current_.view();
}

void StringKeyJoin::close() {
    if (opened_) probe_->close();
    state_.reset();
    cur_probe_.reset();
    opened_ = false;
    probe_done_ = false;
    pair_cursor_ = 0;
}

}  // namespace qe::mutant
