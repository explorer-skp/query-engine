//  WP-4: INTERNAL normalization + per-batch key prep shared by the real
//  HashTable (ops/hashtable.cpp) AND the test-only mutant table
//  (ops/hashtable_mutants.cpp). NOT a frozen contract (the analog of
//  expr/eval_internal.h). Centralizing it here is what lets the mutant be a
//  faithful "real probe minus one step": the mutant reuses this UNCHANGED
//  normalization/hash/equality and breaks ONLY the slot/grow logic, so the
//  reference cross-check isolates the planted probe bug.
//
//  NORMALIZATION = the documented key-equality contract made concrete:
//   * I32  -> zero-extend the 32-bit pattern into a u64 word (low 32 bits recover
//     the value).
//   * I64/TS -> bit_cast<u64>.
//   * BOOL -> the 0/1 byte as a u64.
//   * F64  -> bit_cast<u64> AFTER canonicalizing: any zero (incl. -0.0) -> +0.0
//     bits; any NaN -> one canonical quiet NaN. This is what makes -0.0==+0.0 and
//     NaN==NaN as keys (matching DuckDB), with no float compare in the hot path.
//   * NULL value (per validity) -> a fixed sentinel word kNullWord, AND the row's
//     null-mask bit for that column is set. Correctness never relies on the
//     sentinel being collision-free: key_equal compares null masks first, so a
//     real value that happens to equal kNullWord is still distinguished from a
//     NULL by the mask.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/column.h"
#include "core/types.h"
#include "ops/hashtable.h"  // KeyColumns, HashPath, kDefaultHashSeed

namespace qe::ops::detail {

// Arbitrary fixed sentinel hashed in for NULL lanes (its exact value is
// irrelevant to correctness — see header note).
inline constexpr std::uint64_t kNullWord = 0x9E37'79B9'7F4A'7C15ull;

// Maximum key columns: the per-row/per-group null mask is one u64 bit per column.
inline constexpr std::size_t kMaxKeyColumns = 63;

// Normalize one in-range value of type `t` at byte pointer `p` to its canonical
// key word (the NON-null case; NULL lanes are handled by the caller).
std::uint64_t normalize_value(Type t, const std::byte* p);

// Normalize + hash a whole batch of `n` logical rows of `keys` (key types in
// `types`, table `seed`, kVector/kScalar `path`). Fills:
//   words[j][k]   — canonical key word of column j, row k (kNullWord if null)
//   null_mask[k]  — bit j set iff column j of row k is NULL
//   hash[k]       — combined splitmix64 row hash (seeded), via the chosen path
// The vectors are resized as needed (caller may reuse them as scratch). The hash
// is produced by ops/hash_kernels.h (the vector/scalar twin), so running this
// with kVector vs kScalar is exactly the end-to-end scalar==vector differential.
void normalize_and_hash(const KeyColumns& keys, std::size_t n,
                        const std::vector<Type>& types, std::uint64_t seed,
                        HashPath path,
                        std::vector<std::vector<std::uint64_t>>& words,
                        std::vector<std::uint64_t>& null_mask,
                        std::vector<std::uint64_t>& hash);

// True iff logical row k of a normalized batch (words/null_mask) equals stored
// group `g` (column-major store_words + per-group store_null_mask), per the
// kEqual rule: masks must match, then every NON-null column word must match.
inline bool key_equal(const std::vector<std::vector<std::uint64_t>>& words,
                      const std::vector<std::uint64_t>& null_mask, std::size_t k,
                      const std::vector<std::vector<std::uint64_t>>& store_words,
                      const std::vector<std::uint64_t>& store_null_mask,
                      std::uint32_t g, std::size_t num_cols) {
    const std::uint64_t m = null_mask[k];
    if (m != store_null_mask[g]) return false;
    for (std::size_t j = 0; j < num_cols; ++j) {
        if ((m >> j) & 1ull) continue;  // both NULL here (masks equal) -> skip
        if (words[j][k] != store_words[j][g]) return false;
    }
    return true;
}

}  // namespace qe::ops::detail
