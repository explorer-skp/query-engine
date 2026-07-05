//  WP-5 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken AGGREGATION operators
//  (ops/aggregate_mutants.{h,cpp}) — each a faithful copy of the real grouped
//  accumulate/finalize/output loop with exactly one bad step — and SHOW the
//  differential (independent reference + DuckDB when staged) CATCHES every
//  mutant, while the REAL operator PASSES the identical diff.
//
//  Planted mutants:
//    * kFoldNullInSum     — the §5 named bug: SUM/AVG fold a NULL input as 0 AND
//      count it, so an all-null group emits 0 (and a non-NULL "seen") not NULL.
//    * kEmptyGroupZero    — a zero-non-null group's SUM/MIN/MAX/AVG emits a
//      concrete value (0 / identity) instead of NULL.
//    * kGroupTailOffByOne — output batching drops the LAST group (short final
//      batch -> row-count mismatch).
//
//  Replay:  ./agg_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/aggregate_mutants.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

// c0 key i32 = three groups {1,2,3}; c1 value i32 with group 1 ALL NULL.
struct Case {
    Table table;
    LogicalQuery query;
};

Case make_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I32);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 2, 2, 3, 3}));
    cols.push_back(i32_col({0, 0, 5, 6, 7, 8}, /*nulls=*/{0, 1}));  // group1 null
    LogicalQuery q;
    q.group_by = GroupBy{{0},
                         {AggSpec::count_star("cs"), AggSpec::count(1, "c1"),
                          AggSpec::sum(1, "s1"), AggSpec::min(1, "mn"),
                          AggSpec::max(1, "mx"), AggSpec::avg(1, "av")}};
    return Case{Table(s, std::move(cols)), std::move(q)};
}

DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_reference(c.table, c.query);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            const ResultSet dk = run_duckdb(c.table, c.query);
            d = compare_result_sets(engine, dk);
        } catch (const DuckDBError& e) {
            // Grammar-safe case: a raise is a renderer/oracle regression, not a
            // divergence. Silently skipping would disable DuckDB coverage with
            // CI green (audit C3) -- fail loudly instead.
            FAIL("DuckDB raised on a grammar-safe case: " << std::string(e.what()));
        }
    }
    return d;
}

DiffResult run_mutant(const Case& c, mutant::Mutation m, std::size_t bs) {
    auto scan = std::make_unique<Scan>(c.table, bs);
    mutant::Aggregate agg(std::move(scan), c.query.group_by->keys,
                          c.query.group_by->aggs, m);
    return diff_tree_vs_oracles(agg, c);
}

}  // namespace

TEST_CASE("MUTATION: the REAL aggregate passes the differential") {
    const Case c = make_case();
    for (std::size_t bs : {64u, 2048u})
        CHECK(run_vs_reference(c.table, c.query, bs).equal);
}

TEST_CASE("MUTATION: kFoldNullInSum (NULL folded as 0 in SUM) is CAUGHT") {
    const Case c = make_case();
    const DiffResult d = run_mutant(c, mutant::Mutation::kFoldNullInSum, 64);
    CHECK_FALSE(d.equal);
    MESSAGE("kFoldNullInSum caught: " << d.message);
}

TEST_CASE("MUTATION: kEmptyGroupZero (all-null group emits 0 not NULL) is CAUGHT") {
    const Case c = make_case();
    const DiffResult d = run_mutant(c, mutant::Mutation::kEmptyGroupZero, 64);
    CHECK_FALSE(d.equal);
    MESSAGE("kEmptyGroupZero caught: " << d.message);
}

TEST_CASE("MUTATION: kGroupTailOffByOne (dropped last group) is CAUGHT") {
    const Case c = make_case();
    const DiffResult d = run_mutant(c, mutant::Mutation::kGroupTailOffByOne, 64);
    CHECK_FALSE(d.equal);
    MESSAGE("kGroupTailOffByOne caught: " << d.message);
}
