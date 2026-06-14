//  WP-4: implementation of the TEST-ONLY mutant hash table
//  (ops/hashtable_mutants.h). A line-for-line copy of ops/hashtable.cpp's
//  probe/insert/grow, reusing the SAME ops/hash_internal.h helpers, with exactly
//  one planted defect per Mutation. The `// BUG:` lines mark each deviation from
//  the real code.

#include "ops/hashtable_mutants.h"

#include <cassert>
#include <utility>

#include "ops/hash_internal.h"

namespace qe::mutant {

namespace {
std::size_t round_up_pow2(std::size_t v, std::size_t floor) {
    std::size_t c = floor;
    while (c < v) c <<= 1;
    return c;
}
}  // namespace

HashTable::HashTable(std::vector<Type> key_types, NullPolicy null_policy,
                     Mutation mut, HashTableConfig cfg, std::uint64_t seed)
    : key_types_(std::move(key_types)),
      null_policy_(null_policy),
      mut_(mut),
      seed_(seed) {
    double lf = cfg.max_load_factor;
    if (!(lf > 0.0)) lf = 0.7;
    if (lf > HashTableConfig::kMaxLoadCeiling) lf = HashTableConfig::kMaxLoadCeiling;
    max_load_factor_ = lf;
    capacity_ = round_up_pow2(cfg.initial_capacity, HashTableConfig::kMinCapacity);
    mask_ = capacity_ - 1;
    grow_threshold_ = static_cast<std::size_t>(
        static_cast<double>(capacity_) * max_load_factor_);
    slots_.assign(capacity_, kNoGroup);
    key_words_.resize(key_types_.size());
}

std::uint32_t HashTable::new_group_from(std::size_t k) {
    const std::uint32_t g = next_id_++;
    for (std::size_t j = 0; j < key_words_.size(); ++j)
        key_words_[j].push_back(scratch_words_[j][k]);
    group_null_mask_.push_back(scratch_null_[k]);
    group_hash_.push_back(scratch_hash_[k]);
    return g;
}

void HashTable::grow() {
    const std::size_t new_cap = capacity_ * 2;
    const std::size_t new_mask = new_cap - 1;
    std::vector<std::uint32_t> ns(new_cap, kNoGroup);
    bool skipped = false;
    for (std::size_t s = 0; s < capacity_; ++s) {
        const std::uint32_t g = slots_[s];
        if (g == kNoGroup) continue;
        if (mut_ == Mutation::kSkipRehashOne && !skipped) {
            // BUG: drop the first element instead of re-slotting it.
            skipped = true;
            continue;
        }
        std::size_t slot = group_hash_[g] & new_mask;
        while (ns[slot] != kNoGroup) slot = (slot + 1) & new_mask;
        ns[slot] = g;
    }
    slots_.swap(ns);
    capacity_ = new_cap;
    mask_ = new_mask;
    grow_threshold_ = static_cast<std::size_t>(
        static_cast<double>(capacity_) * max_load_factor_);
}

void HashTable::insert_or_find(const KeyColumns& keys, std::size_t n,
                               std::uint32_t* out_groups, HashPath path) {
    ops::detail::normalize_and_hash(keys, n, key_types_, seed_, path,
                                    scratch_words_, scratch_null_, scratch_hash_);
    const std::size_t ncols = key_types_.size();
    for (std::size_t k = 0; k < n; ++k) {
        if (null_policy_ == NullPolicy::kNeverMatch && scratch_null_[k] != 0) {
            out_groups[k] = new_group_from(k);
            continue;
        }
        const std::uint64_t h = scratch_hash_[k];
        std::size_t slot = h & mask_;
        std::uint32_t found = kNoGroup;
        while (slots_[slot] != kNoGroup) {
            const std::uint32_t g = slots_[slot];
            if (group_hash_[g] == h &&
                ops::detail::key_equal(scratch_words_, scratch_null_, k,
                                       key_words_, group_null_mask_, g, ncols)) {
                found = g;
                break;
            }
            slot = (slot + 1) & mask_;
        }
        if (found == kNoGroup) {
            const std::uint32_t g = new_group_from(k);
            slots_[slot] = g;
            ++occupied_;
            found = g;
            if (occupied_ > grow_threshold_) grow();
        }
        out_groups[k] = found;
    }
}

void HashTable::find(const KeyColumns& keys, std::size_t n,
                     std::uint32_t* out_groups, HashPath path) const {
    ops::detail::normalize_and_hash(keys, n, key_types_, seed_, path,
                                    scratch_words_, scratch_null_, scratch_hash_);
    const std::size_t ncols = key_types_.size();
    for (std::size_t k = 0; k < n; ++k) {
        if (null_policy_ == NullPolicy::kNeverMatch && scratch_null_[k] != 0) {
            out_groups[k] = kNoGroup;
            continue;
        }
        const std::uint64_t h = scratch_hash_[k];
        std::size_t slot = h & mask_;
        std::uint32_t found = kNoGroup;
        while (slots_[slot] != kNoGroup) {
            const std::uint32_t g = slots_[slot];
            if (group_hash_[g] == h &&
                ops::detail::key_equal(scratch_words_, scratch_null_, k,
                                       key_words_, group_null_mask_, g, ncols)) {
                found = g;
                break;
            }
            slot = (slot + 1) & mask_;
            if (mut_ == Mutation::kFindStopEarly) {
                // BUG: give up after the first probed slot instead of walking
                // the whole collision chain.
                break;
            }
        }
        out_groups[k] = found;
    }
}

}  // namespace qe::mutant
