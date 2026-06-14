//  WP-4: the shared HASH TABLE SUBSTRATE — open-addressing, linear-probe,
//  power-of-two capacity, vectorized group-find/probe (decision D8). This is the
//  ground both WP-5 (hash aggregation / GROUP BY) and WP-6 (hash join) build on:
//  it groups/probes rows by one or more key columns at vector speed and hands
//  back a stable, dense GROUP ID per distinct key.
//
//  PROPOSED FROZEN CONTRACT (freeze at WP-4 acceptance). The PUBLIC SURFACE below
//  — the value types (NullPolicy, HashTableConfig, KeyColumns, HashPath), the
//  constants (kNoGroup, kDefaultHashSeed), and the HashTable class signature — is
//  what WP-5/WP-6 depend on. Later WPs may not add, rename, or retype a public
//  member without an Interface Change Request. (The IMPLEMENTATION internals —
//  the hash kernels, normalization, mutants — live in sibling non-frozen headers.)
//
//  WHAT IT IS / IS NOT. The table is a key->group-id index plus a read-back KEY
//  STORE. It deliberately does NOT store payloads, run aggregates, or gather
//  probe matches — those are WP-5/WP-6's job. It gives them exactly two
//  primitives and a way to read each group's key tuple back:
//    * insert_or_find — the BUILD / GROUP path: every row gets a group id;
//      identical keys collapse to one id, distinct keys get the next id.
//    * find           — the PROBE path: look a key up WITHOUT inserting.
//
//  KEY TYPES (D7): I32, I64, F64, BOOL, TS. Single or COMPOSITE keys (a tuple of
//  columns), composite via hashing the per-column key words together.
//
//  KEY EQUALITY SEMANTICS (documented contract):
//   * I32/I64/TS/BOOL: exact bitwise equality of the value.
//   * F64: equality is on a CANONICALIZED bit pattern, so it matches DuckDB's
//     GROUP BY / join key semantics rather than raw IEEE compare:
//       - -0.0 and +0.0 are the SAME key (both canonicalize to +0.0 bits).
//       - every NaN (any payload/sign) is the SAME key (canonical quiet NaN),
//         i.e. NaN groups/joins with NaN. (Raw `==` says NaN != NaN; we do not.)
//
//  NULL KEYS (we do NOT bake one SQL policy in; the caller picks per table — see
//  NullPolicy — so WP-5 and WP-6 each get the semantics SQL demands):
//   * NullPolicy::kEqual    — a NULL is a value: NULL==NULL (same column
//     position), NULL!=non-null. Null-bearing tuples form findable groups. This
//     is GROUP BY semantics (all NULLs group together).
//   * NullPolicy::kNeverMatch — a key tuple containing ANY NULL matches nothing,
//     not even an identical NULL tuple. find() of such a key MISSES; an
//     insert_or_find() of such a key gets a FRESH, unique, UNFINDABLE ("dead")
//     group id every time (never deduped, never probeable). This is join key
//     semantics (NULLs never join).
//
//  GROUP IDs ARE STABLE. Ids are assigned 0,1,2,... in first-seen order and never
//  change — not even across a table growth/rehash (growth re-slots groups but
//  preserves their ids). WP-5 indexes its per-group aggregate-state arrays by id
//  and relies on this.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/column.h"
#include "core/types.h"

namespace qe {

// "Not a group": returned by find() on a miss, and used internally as the
// empty-slot marker. The maximum number of distinct groups is therefore
// kNoGroup (ids 0..kNoGroup-1); exceeding it is a guarded error, not silent
// wraparound.
inline constexpr std::uint32_t kNoGroup = UINT32_MAX;

// Default hash seed. Randomized tests override it via the constructor and PRINT
// the seed (RIGOR.md rule 5), so a fuzz failure replays deterministically.
inline constexpr std::uint64_t kDefaultHashSeed = 0x51E5'7A11'AB1E'5EEDull;

// How NULL key values participate in equality. See the header comment.
enum class NullPolicy { kEqual, kNeverMatch };

// Which code path computes the bulk key hashes. The probe/slot walk is shared
// scalar control flow either way; only the (vectorizable) hash kernel differs.
// Production uses kVector; the scalar==vector differential drives both and
// asserts identical group ids (RIGOR.md rule 3 / D17). Exposed so that test is
// a true end-to-end check; callers normally take the default.
enum class HashPath { kVector, kScalar };

// Growth policy — named config, NOT magic constants buried in the body
// (RIGOR.md: never hardcode capacity / load factor). Both are validated and
// normalized at construction: initial_capacity is rounded UP to a power of two
// (floored at kMinCapacity); max_load_factor is clamped to (0, kMaxLoadCeiling].
struct HashTableConfig {
    // Initial slot count. Rounded up to the next power of two at runtime.
    std::size_t initial_capacity = 1024;
    // Grow (double capacity) once occupied slots exceed capacity*max_load_factor.
    // < 1.0 guarantees a free slot always exists, so the probe never loops
    // forever (the §12 hash-probe-overflow hazard).
    double max_load_factor = 0.7;

    // Floors/ceilings the above are normalized against (named, not magic).
    static constexpr std::size_t kMinCapacity = 8;       // smallest power of two
    static constexpr double kMaxLoadCeiling = 0.95;      // never fill past this
};

// A view of the KEY COLUMNS of a batch, in key order, plus the batch's optional
// selection vector. `cols` points at `num_cols` Column views (typically a small
// caller-built array selecting the key columns out of a wider Batch). All
// columns share the same physical layout and the same selection; logical row k
// (k in [0,n)) reads physical row sel_at(sel, k). For a single-key table
// num_cols == 1; for a composite key it is the tuple width.
struct KeyColumns {
    const Column* cols;
    std::size_t num_cols;
    const SelectionVector* sel;  // nullptr => dense (logical k == physical k)
};

class HashTable {
   public:
    // Construct over a key schema (`key_types`, in key order; 1..63 columns —
    // the per-row null mask is a uint64 bit per column). `null_policy` picks the
    // NULL semantics (see header). `cfg` is the growth policy. `seed` seeds the
    // mixing hash (printed by the test harness for replay).
    explicit HashTable(std::vector<Type> key_types,
                       NullPolicy null_policy = NullPolicy::kEqual,
                       HashTableConfig cfg = {},
                       std::uint64_t seed = kDefaultHashSeed);

    // INSERT-OR-FIND for `n` logical rows of `keys`. Writes out_groups[k] = the
    // stable group id of row k's key tuple: identical keys (per policy) get the
    // same id; a new distinct key gets the next sequential id. Grows as needed.
    // out_groups must have room for n ids. (Never returns kNoGroup.)
    void insert_or_find(const KeyColumns& keys, std::size_t n,
                        std::uint32_t* out_groups,
                        HashPath path = HashPath::kVector);

    // FIND-ONLY (the join probe). Writes out_groups[k] = the group id of an
    // existing matching key, or kNoGroup if absent. NEVER mutates the table
    // (const). out_groups must have room for n ids.
    void find(const KeyColumns& keys, std::size_t n, std::uint32_t* out_groups,
              HashPath path = HashPath::kVector) const;

    // Number of distinct group ids assigned so far (== the next id to assign).
    std::size_t num_groups() const noexcept { return next_id_; }
    // Current slot-array size (a power of two).
    std::size_t capacity() const noexcept { return capacity_; }
    NullPolicy null_policy() const noexcept { return null_policy_; }
    std::uint64_t seed() const noexcept { return seed_; }
    const std::vector<Type>& key_types() const noexcept { return key_types_; }

    // --- KEY STORE READ-BACK (so WP-5/WP-6 can materialize group keys) --------
    // True iff column `col` of group `g` is NULL. Precondition: g < num_groups().
    bool group_is_null(std::uint32_t g, std::size_t col) const {
        return (group_null_mask_[g] >> col) & 1ull;
    }
    // The CANONICAL 64-bit key word stored for column `col` of group `g`. Decode
    // by the column's Type (the canonicalization is the one documented above):
    //   I32  -> static_cast<int32_t>(word)
    //   I64/TS -> bit_cast<int64_t>(word)
    //   F64  -> bit_cast<double>(word)   (+0.0 for any zero; canonical NaN)
    //   BOOL -> static_cast<uint8_t>(word) (0/1)
    // Only meaningful where !group_is_null(g, col). Precondition: g<num_groups().
    std::uint64_t group_key_word(std::uint32_t g, std::size_t col) const {
        return key_words_[col][g];
    }

   private:
    void grow();
    // Append a brand-new group built from logical row k of the normalized batch;
    // returns its id. Does NOT touch the slot array (callers slot it, or leave it
    // "dead" for kNeverMatch null keys).
    std::uint32_t new_group_from(std::size_t k);

    std::vector<Type> key_types_;
    NullPolicy null_policy_;
    std::uint64_t seed_;
    double max_load_factor_;

    std::size_t capacity_;        // power of two
    std::size_t mask_;            // capacity_ - 1
    std::size_t occupied_ = 0;    // groups currently in the slot array
    std::size_t grow_threshold_;  // capacity_ * max_load_factor_
    std::uint32_t next_id_ = 0;   // next group id to assign

    std::vector<std::uint32_t> slots_;  // capacity_ entries; kNoGroup == empty

    // Key store, indexed by group id. Column-major words + per-group null mask +
    // per-group hash (so growth re-slots without recomputing, and probes compare
    // the cheap hash before the full key).
    std::vector<std::vector<std::uint64_t>> key_words_;  // [num_cols][group]
    std::vector<std::uint64_t> group_null_mask_;         // [group]
    std::vector<std::uint64_t> group_hash_;              // [group]

    // Scratch reused across calls: the normalized current batch. Populated by
    // insert_or_find/find via the shared normalization (ops/hash_internal.h).
    // `find` is logically const but fills scratch, hence mutable.
    mutable std::vector<std::vector<std::uint64_t>> scratch_words_;
    mutable std::vector<std::uint64_t> scratch_null_;
    mutable std::vector<std::uint64_t> scratch_hash_;
};

}  // namespace qe
