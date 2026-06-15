//  WP-9 catalog checks — hash-table probe/growth hazards. Includes ONLY
//  ops/hashtable_mutants.h among the three conflicting qe::mutant::Mutation
//  headers, so it gets its own TU. The "checker" is the same collision-storm
//  cross-check the WP-4 fuzz/mutation suite uses: distinct sequential keys in a
//  tiny, densely packed table force many growths and long probe chains; the truth
//  is "all keys distinct => ids 0..n-1, every key findable".

#include "tests/catalog_checks.h"

#include <cstdint>
#include <vector>

#include "core/types.h"
#include "ops/hashtable.h"
#include "ops/hashtable_mutants.h"
#include "tests/hashtable_test_util.h"

namespace qe::catalog::checks {

using namespace qe;

namespace {

// A fixed hash seed keeps the meta-test reproducible; the collision storm forms
// regardless of seed (sequential keys, tiny capacity, 0.9 load factor).
constexpr std::uint64_t kCatalogSeed = 0x9E3779B97F4A7C15ull;

HashTableConfig dense_cfg() {
    HashTableConfig c;
    c.initial_capacity = 8;   // force many doublings
    c.max_load_factor = 0.9;  // pack densely -> long probe chains
    return c;
}

template <typename Table_>
bool agrees_with_reference(Table_& t, std::size_t n) {
    std::vector<std::int64_t> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = static_cast<std::int64_t>(i);
    ht_test::KeyHolder h;
    h.owned.push_back(ht_test::make_col(Type::I64, keys));
    const KeyColumns kc = h.kc();

    std::vector<std::uint32_t> g(n);
    t.insert_or_find(kc, n, g.data());
    for (std::size_t i = 0; i < n; ++i)
        if (g[i] != static_cast<std::uint32_t>(i)) return false;
    if (t.num_groups() != n) return false;
    std::vector<std::uint32_t> fg(n);
    t.find(kc, n, fg.data());
    for (std::size_t i = 0; i < n; ++i)
        if (fg[i] != static_cast<std::uint32_t>(i)) return false;
    return true;
}

}  // namespace

Verdict hash_probe_overflow() {
    HashTable real({Type::I64}, NullPolicy::kEqual, dense_cfg(), kCatalogSeed);
    mutant::HashTable bad({Type::I64}, NullPolicy::kEqual,
                          mutant::Mutation::kFindStopEarly, dense_cfg(),
                          kCatalogSeed);
    Verdict v;
    v.clean_passes = agrees_with_reference(real, 4000);
    v.mutant_flagged = !agrees_with_reference(bad, 4000);
    v.detail = "kFindStopEarly: probe gives up one slot early on a collision storm";
    return v;
}

Verdict hash_growth_rehash() {
    HashTable real({Type::I64}, NullPolicy::kEqual, dense_cfg(), kCatalogSeed);
    mutant::HashTable bad({Type::I64}, NullPolicy::kEqual,
                          mutant::Mutation::kSkipRehashOne, dense_cfg(),
                          kCatalogSeed);
    Verdict v;
    v.clean_passes = agrees_with_reference(real, 4000);
    v.mutant_flagged = !agrees_with_reference(bad, 4000);
    v.detail = "kSkipRehashOne: one group not re-slotted on growth => unfindable";
    return v;
}

}  // namespace qe::catalog::checks
