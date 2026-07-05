//  WP-4: HashTable implementation (see ops/hashtable.h for the frozen contract).
//
//  Open-addressing, linear probing, power-of-two capacity (decision D8). The
//  slot array maps slot -> group id (or kNoGroup == empty). The key store
//  (column-major words + per-group null mask + per-group hash), indexed by group
//  id, lets a probe compare the cheap stored hash before the full key tuple, and
//  lets growth re-slot every group from its stored hash without recomputing it.
//
//  The data-parallel work (the from-scratch mixing hash over the key columns)
//  goes through the vector/scalar twin (ops/hash_kernels.h via
//  ops/hash_internal.h). The slot/probe WALK below is inherently sequential
//  per row — pointer-chasing with data-dependent probe length — so it is shared
//  scalar control flow, NOT vectorized; its correctness is pinned by the
//  reference-model fuzz cross-check (the §12 hash-probe-overflow / growth
//  hotspot), and the planted-probe mutation self-test shows that check bites.

#include "ops/hashtable.h"

#include <cassert>
#include <stdexcept>

#include "ops/hash_internal.h"

namespace qe {

namespace {
// Smallest power of two >= max(v, floor). `floor` must itself be a power of two.
std::size_t round_up_pow2(std::size_t v, std::size_t floor) {
    std::size_t c = floor;
    while (c < v) c <<= 1;
    return c;
}
}  // namespace

HashTable::HashTable(std::vector<Type> key_types, NullPolicy null_policy,
                     HashTableConfig cfg, std::uint64_t seed)
    : key_types_(std::move(key_types)),
      null_policy_(null_policy),
      seed_(seed) {
    assert(!key_types_.empty() &&
           key_types_.size() <= ops::detail::kMaxKeyColumns &&
           "key column count must be in [1, 63]");

    // Normalize the growth policy — clamp/round here so no magic leaks into the
    // hot path and the invariant (load factor < 1 => a free slot always exists)
    // is guaranteed.
    double lf = cfg.max_load_factor;
    if (!(lf > 0.0)) lf = 0.7;  // also rejects NaN
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
    // Real guard in EVERY build (audit H5): hashtable.h promises "a guarded
    // error, not silent wraparound", but an assert vanishes under NDEBUG —
    // and a wrapped id would alias group 0's state or masquerade as kNoGroup.
    if (next_id_ == kNoGroup)
        throw std::overflow_error("HashTable: group id space exhausted (2^32-1)");
    const std::uint32_t g = next_id_++;
    for (std::size_t j = 0; j < key_words_.size(); ++j) {
        key_words_[j].push_back(scratch_words_[j][k]);
    }
    group_null_mask_.push_back(scratch_null_[k]);
    group_hash_.push_back(scratch_hash_[k]);
    return g;
}

void HashTable::grow() {
    const std::size_t new_cap = capacity_ * 2;
    const std::size_t new_mask = new_cap - 1;
    std::vector<std::uint32_t> ns(new_cap, kNoGroup);
    for (std::size_t s = 0; s < capacity_; ++s) {
        const std::uint32_t g = slots_[s];
        if (g == kNoGroup) continue;
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
    assert(keys.num_cols == key_types_.size() && "key arity mismatch");
    ops::detail::normalize_and_hash(keys, n, key_types_, seed_, path,
                                    scratch_words_, scratch_null_, scratch_hash_);

    const std::size_t ncols = key_types_.size();
    for (std::size_t k = 0; k < n; ++k) {
        // kNeverMatch: a key tuple with ANY null is unmatchable. Give it a fresh
        // UNFINDABLE ("dead") group id — stored for read-back, never slotted, so
        // it is never deduped and a later find() can never reach it.
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
            // Grow AFTER inserting, keeping load < 1 so a free slot always
            // remains (the probe above can never loop forever).
            if (occupied_ > grow_threshold_) grow();
        }
        out_groups[k] = found;
    }
}

void HashTable::find(const KeyColumns& keys, std::size_t n,
                     std::uint32_t* out_groups, HashPath path) const {
    assert(keys.num_cols == key_types_.size() && "key arity mismatch");
    ops::detail::normalize_and_hash(keys, n, key_types_, seed_, path,
                                    scratch_words_, scratch_null_, scratch_hash_);

    const std::size_t ncols = key_types_.size();
    for (std::size_t k = 0; k < n; ++k) {
        // kNeverMatch: a null-bearing probe key matches nothing.
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
        }
        out_groups[k] = found;
    }
}

}  // namespace qe
