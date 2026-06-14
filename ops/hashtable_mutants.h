//  WP-4: TEST-ONLY planted-mutant hash table. "A checker that cannot fail proves
//  nothing." (RIGOR.md rule 4.)
//
//  qe::mutant::HashTable is a faithful copy of the real probe/insert/grow loops
//  (ops/hashtable.cpp) that REUSES the exact same shared normalization, hashing
//  and key-equality (ops/hash_internal.h) — so the ONLY difference from the real
//  table is one deliberately-broken step in the slot/grow logic, selected by the
//  `Mutation` enum. The mutation self-test (tests/hashtable_mutation_test.cpp)
//  runs the real table and a mutant through the SAME std::unordered_map reference
//  cross-check and shows the check CATCHES each mutant while the real table
//  passes — the §12 hash-probe-overflow / capacity-growth hotspot.
//
//  This is NOT linked into any engine target — only the mutation test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/types.h"
#include "ops/hashtable.h"  // KeyColumns, NullPolicy, HashTableConfig, HashPath

namespace qe::mutant {

enum class Mutation {
    // BUG: on growth, the FIRST occupied group is not re-slotted, so it becomes
    // unreachable by find() after the table grows. The reference model still
    // reports it present -> caught. (The "skip rehash of one element" mutation.)
    kSkipRehashOne,
    // BUG: find() inspects only the FIRST probed slot and gives up if it is not a
    // match (stops probing one slot too early), so any key displaced by a
    // collision is reported missing -> caught on collision storms.
    kFindStopEarly,
};

// Mirror of qe::HashTable's relevant surface, with a planted defect.
class HashTable {
   public:
    HashTable(std::vector<Type> key_types, NullPolicy null_policy, Mutation mut,
              HashTableConfig cfg = {},
              std::uint64_t seed = kDefaultHashSeed);

    void insert_or_find(const KeyColumns& keys, std::size_t n,
                        std::uint32_t* out_groups,
                        HashPath path = HashPath::kVector);
    void find(const KeyColumns& keys, std::size_t n, std::uint32_t* out_groups,
              HashPath path = HashPath::kVector) const;

    std::size_t num_groups() const noexcept { return next_id_; }
    std::size_t capacity() const noexcept { return capacity_; }

   private:
    void grow();
    std::uint32_t new_group_from(std::size_t k);

    std::vector<Type> key_types_;
    NullPolicy null_policy_;
    Mutation mut_;
    std::uint64_t seed_;
    double max_load_factor_;
    std::size_t capacity_;
    std::size_t mask_;
    std::size_t occupied_ = 0;
    std::size_t grow_threshold_;
    std::uint32_t next_id_ = 0;
    std::vector<std::uint32_t> slots_;
    std::vector<std::vector<std::uint64_t>> key_words_;
    std::vector<std::uint64_t> group_null_mask_;
    std::vector<std::uint64_t> group_hash_;
    mutable std::vector<std::vector<std::uint64_t>> scratch_words_;
    mutable std::vector<std::uint64_t> scratch_null_;
    mutable std::vector<std::uint64_t> scratch_hash_;
};

}  // namespace qe::mutant
