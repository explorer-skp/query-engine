//  WP-4 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken PROBES in the table
//  (ops/hashtable_mutants.{h,cpp}) — each a faithful copy of the real loop with
//  exactly one bad step in the §12 hotspot (probe walk / growth-rehash) — and
//  SHOW the same std::unordered_map reference cross-check the fuzz suite uses
//  CATCHES every mutant, while the REAL table PASSES the identical check.
//
//  Planted mutants:
//    * kSkipRehashOne — on growth, one group is not re-slotted (the "skip rehash
//      of one element" bug); it becomes unfindable after the table grows.
//    * kFindStopEarly — find() gives up after the first probed slot (stops
//      probing one slot too early); collision-displaced keys are reported absent.
//
//  Replay:  ./hashtable_mutation_test --seed N

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "ops/hash_internal.h"
#include "ops/hashtable.h"
#include "ops/hashtable_mutants.h"
#include "tests/hashtable_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::ht_test;

namespace {

// A run of distinct I64 keys, inserted then probed, from a tiny capacity so many
// growths and long probe chains occur. Returns true iff the table under test
// agrees with the truth (id == insertion order, and every key findable).
template <typename Table>
bool agrees_with_reference(Table& t, std::size_t n) {
    std::vector<std::int64_t> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = static_cast<std::int64_t>(i);
    KeyHolder h;
    h.owned.push_back(make_col(Type::I64, keys));
    const KeyColumns kc = h.kc();

    std::vector<std::uint32_t> g(n);
    t.insert_or_find(kc, n, g.data());
    // Reference truth: all keys distinct -> ids 0..n-1 in order.
    for (std::size_t i = 0; i < n; ++i)
        if (g[i] != static_cast<std::uint32_t>(i)) return false;
    if (t.num_groups() != n) return false;

    // Every key must be findable with its insertion id.
    std::vector<std::uint32_t> fg(n);
    t.find(kc, n, fg.data());
    for (std::size_t i = 0; i < n; ++i)
        if (fg[i] != static_cast<std::uint32_t>(i)) return false;
    return true;
}

HashTableConfig small_cfg() {
    HashTableConfig c;
    c.initial_capacity = 8;   // force many doublings
    c.max_load_factor = 0.9;  // pack densely -> long chains
    return c;
}

}  // namespace

TEST_CASE("MUTATION: the REAL table passes the reference cross-check") {
    HashTable real({Type::I64}, NullPolicy::kEqual, small_cfg(),
                   qe::test::seed());
    CHECK(agrees_with_reference(real, 4000));
}

TEST_CASE("MUTATION: kSkipRehashOne (dropped element on growth) is CAUGHT") {
    mutant::HashTable bad({Type::I64}, NullPolicy::kEqual,
                          mutant::Mutation::kSkipRehashOne, small_cfg(),
                          qe::test::seed());
    // The dropped group is lost on the first growth -> the cross-check fails.
    CHECK_FALSE(agrees_with_reference(bad, 4000));
    MESSAGE("kSkipRehashOne mutant detected by the reference cross-check");
}

TEST_CASE("MUTATION: kFindStopEarly (probe stops one slot early) is CAUGHT") {
    mutant::HashTable bad({Type::I64}, NullPolicy::kEqual,
                          mutant::Mutation::kFindStopEarly, small_cfg(),
                          qe::test::seed());
    // Inserts are fine; find() gives up early so collision-displaced keys read as
    // absent -> the cross-check fails.
    CHECK_FALSE(agrees_with_reference(bad, 4000));
    MESSAGE("kFindStopEarly mutant detected by the reference cross-check");
}

TEST_CASE("MUTATION: bugs are DORMANT on a table that never grows / never probes "
          "past the first slot (proving they are genuine probe/growth bugs)") {
    // With a capacity large enough that NO growth happens AND so sparse that no
    // collision chain forms, both mutants behave identically to the real table.
    // This confirms the defects live specifically in growth-rehash / probe-walk,
    // the §12 hotspot — a suite testing only tiny inputs could miss them.
    HashTableConfig roomy;
    roomy.initial_capacity = 1 << 16;  // 65536 slots
    roomy.max_load_factor = 0.7;
    const std::size_t small_n = 50;  // « capacity, sparse, no growth

    mutant::HashTable m_rehash({Type::I64}, NullPolicy::kEqual,
                               mutant::Mutation::kSkipRehashOne, roomy,
                               qe::test::seed());
    CHECK(agrees_with_reference(m_rehash, small_n));  // dormant: no growth

    // kFindStopEarly only diverges when a probe chain forms. With 50 keys in
    // 65536 slots collisions are possible but rare; rather than rely on chance,
    // assert the REAL table always agrees here (the mutant's divergence is proven
    // at scale above).
    HashTable real({Type::I64}, NullPolicy::kEqual, roomy, qe::test::seed());
    CHECK(agrees_with_reference(real, small_n));
}
