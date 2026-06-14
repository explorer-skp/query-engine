//  WP-4 UNIT tests for the hash table substrate. Deterministic, value-level
//  checks of the contract (ops/hashtable.h):
//    * insert-or-find returns identical ids for identical keys; distinct keys get
//      distinct, dense ids.
//    * find-without-insert misses correctly, and does not mutate the table.
//    * composite (multi-column) keys.
//    * NULL handling under BOTH policies (kEqual groups nulls; kNeverMatch makes
//      null keys unmatchable).
//    * F64 -0.0/+0.0 collapse and NaN==NaN as keys.
//    * key-store read-back (group_key_word / group_is_null).
//  Replay:  ./hashtable_test --seed N   (these cases are deterministic.)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "ops/hashtable.h"
#include "tests/hashtable_test_util.h"

using namespace qe;
using namespace qe::ht_test;

namespace {
std::vector<std::uint32_t> ids(std::size_t n) { return std::vector<std::uint32_t>(n); }
}  // namespace

TEST_CASE("insert_or_find: identical keys -> same id, distinct -> dense ids") {
    HashTable t({Type::I32});
    KeyHolder h;
    h.owned.push_back(make_col(Type::I32, {7, 7, 3, 7, 3, 100}));
    auto g = ids(6);
    t.insert_or_find(h.kc(), 6, g.data());

    CHECK(g[0] == g[1]);          // 7 == 7
    CHECK(g[0] == g[3]);          // 7 == 7
    CHECK(g[2] == g[4]);          // 3 == 3
    CHECK(g[0] != g[2]);          // 7 != 3
    CHECK(g[0] != g[5]);          // 7 != 100
    CHECK(t.num_groups() == 3);   // {7,3,100}
    // Dense ids in first-seen order: 7->0, 3->1, 100->2.
    CHECK(g[0] == 0u);
    CHECK(g[2] == 1u);
    CHECK(g[5] == 2u);
}

TEST_CASE("insert_or_find is idempotent across calls; ids stay stable") {
    HashTable t({Type::I64});
    KeyHolder h;
    h.owned.push_back(make_col(Type::I64, {10, 20, 30}));
    auto g1 = ids(3);
    t.insert_or_find(h.kc(), 3, g1.data());
    REQUIRE(t.num_groups() == 3);

    // Re-inserting the same keys returns the SAME ids and adds no groups.
    auto g2 = ids(3);
    t.insert_or_find(h.kc(), 3, g2.data());
    CHECK(g2 == g1);
    CHECK(t.num_groups() == 3);
}

TEST_CASE("find: misses on absent keys, hits on present, never mutates") {
    HashTable t({Type::I32});
    KeyHolder build;
    build.owned.push_back(make_col(Type::I32, {1, 2, 3}));
    auto bg = ids(3);
    t.insert_or_find(build.kc(), 3, bg.data());
    REQUIRE(t.num_groups() == 3);

    KeyHolder probe;
    probe.owned.push_back(make_col(Type::I32, {2, 99, 1, 12345}));
    auto pg = ids(4);
    t.find(probe.kc(), 4, pg.data());
    CHECK(pg[0] == bg[1]);       // 2 present
    CHECK(pg[1] == kNoGroup);    // 99 absent
    CHECK(pg[2] == bg[0]);       // 1 present
    CHECK(pg[3] == kNoGroup);    // absent
    CHECK(t.num_groups() == 3);  // find added nothing
}

TEST_CASE("composite keys: equality is on the whole tuple") {
    HashTable t({Type::I32, Type::I64});
    KeyHolder h;
    // rows: (1,10) (1,10) (1,20) (2,10)
    h.owned.push_back(make_col(Type::I32, {1, 1, 1, 2}));
    h.owned.push_back(make_col(Type::I64, {10, 10, 20, 10}));
    auto g = ids(4);
    t.insert_or_find(h.kc(), 4, g.data());
    CHECK(g[0] == g[1]);        // (1,10)==(1,10)
    CHECK(g[0] != g[2]);        // (1,10)!=(1,20)
    CHECK(g[0] != g[3]);        // (1,10)!=(2,10)
    CHECK(t.num_groups() == 3);
}

TEST_CASE("NULL policy kEqual: NULLs group together, distinct from non-null") {
    HashTable t({Type::I32}, NullPolicy::kEqual);
    KeyHolder h;
    // rows: NULL, 5, NULL, 5, NULL  (indices 0,2,4 null)
    h.owned.push_back(make_col(Type::I32, {0, 5, 0, 5, 0}, {0, 2, 4}));
    auto g = ids(5);
    t.insert_or_find(h.kc(), 5, g.data());
    CHECK(g[0] == g[2]);          // NULL == NULL
    CHECK(g[0] == g[4]);          // NULL == NULL
    CHECK(g[1] == g[3]);          // 5 == 5
    CHECK(g[0] != g[1]);          // NULL != 5
    CHECK(t.num_groups() == 2);   // {NULL, 5}
    // The null group reads back as null.
    CHECK(t.group_is_null(g[0], 0));
    CHECK_FALSE(t.group_is_null(g[1], 0));
}

TEST_CASE("NULL policy kEqual composite: (NULL,5) groups with (NULL,5)") {
    HashTable t({Type::I32, Type::I32}, NullPolicy::kEqual);
    KeyHolder h;
    h.owned.push_back(make_col(Type::I32, {0, 0, 7}, {0, 1}));  // col0: NULL,NULL,7
    h.owned.push_back(make_col(Type::I32, {5, 5, 5}));          // col1: 5,5,5
    auto g = ids(3);
    t.insert_or_find(h.kc(), 3, g.data());
    CHECK(g[0] == g[1]);   // (NULL,5)==(NULL,5)
    CHECK(g[0] != g[2]);   // (NULL,5)!=(7,5)
    CHECK(t.num_groups() == 2);
}

TEST_CASE("NULL policy kNeverMatch: null keys never match (join semantics)") {
    HashTable t({Type::I32}, NullPolicy::kNeverMatch);
    KeyHolder h;
    h.owned.push_back(make_col(Type::I32, {0, 5, 0, 5}, {0, 2}));  // NULL,5,NULL,5
    auto g = ids(4);
    t.insert_or_find(h.kc(), 4, g.data());
    // The two NULL rows get DISTINCT (dead) ids — they never dedup.
    CHECK(g[0] != g[2]);
    // The two 5s still collapse.
    CHECK(g[1] == g[3]);

    // find of a NULL key MISSES even though a null was "inserted".
    KeyHolder probe;
    probe.owned.push_back(make_col(Type::I32, {0, 5}, {0}));  // NULL, 5
    auto pg = ids(2);
    t.find(probe.kc(), 2, pg.data());
    CHECK(pg[0] == kNoGroup);   // NULL probe misses
    CHECK(pg[1] == g[1]);       // 5 hits
}

TEST_CASE("F64 keys: -0.0 == +0.0 and NaN == NaN as keys") {
    HashTable t({Type::F64});
    KeyHolder h;
    const double nan1 = std::nan("1");
    const double nan2 = -std::numeric_limits<double>::quiet_NaN();
    h.owned.push_back(make_col(
        Type::F64, {f64_bits(0.0), f64_bits(-0.0), f64_bits(nan1),
                    f64_bits(nan2), f64_bits(1.5)}));
    auto g = ids(5);
    t.insert_or_find(h.kc(), 5, g.data());
    CHECK(g[0] == g[1]);        // +0.0 == -0.0
    CHECK(g[2] == g[3]);        // NaN == NaN (any payload/sign)
    CHECK(g[0] != g[2]);        // 0.0 != NaN
    CHECK(g[0] != g[4]);        // 0.0 != 1.5
    CHECK(t.num_groups() == 3); // {0.0, NaN, 1.5}
}

TEST_CASE("key-store read-back: group_key_word decodes to the value") {
    HashTable t({Type::I32, Type::F64});
    KeyHolder h;
    h.owned.push_back(make_col(Type::I32, {42}));
    h.owned.push_back(make_col(Type::F64, {f64_bits(3.25)}));
    auto g = ids(1);
    t.insert_or_find(h.kc(), 1, g.data());
    const std::uint32_t id = g[0];
    CHECK(static_cast<std::int32_t>(t.group_key_word(id, 0)) == 42);
    double d;
    std::uint64_t w = t.group_key_word(id, 1);
    std::memcpy(&d, &w, 8);
    CHECK(d == doctest::Approx(3.25));
}

TEST_CASE("selection vector: only selected rows participate") {
    HashTable t({Type::I32});
    KeyHolder h;
    h.owned.push_back(make_col(Type::I32, {9, 9, 9, 8, 8}));
    h.set_selection({0, 3});  // pick physical rows 0 (9) and 3 (8)
    auto g = ids(2);
    t.insert_or_find(h.kc(), 2, g.data());
    CHECK(g[0] != g[1]);
    CHECK(t.num_groups() == 2);
}
