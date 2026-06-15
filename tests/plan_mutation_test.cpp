//  WP-8 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4). We deliberately MIS-LOWER
//  a Plan — one plan-LOWERING defect at a time from the brief's catalog — and SHOW
//  the plan differential (vs the independent plan reference, and vs DuckDB when
//  staged) FLAGS it; the CORRECT Plan::lower() passes the same diff. The defect is
//  in the LOWERING, not the operators (those have their own WP mutation tests), so
//  this proves the new plan/orchestration layer is itself checked.
//
//  Catalog demonstrated (plan/plan_mutants.h):
//    * kDropFilter     — drop a Filter node          (deep pipeline; row mismatch)
//    * kSwapJoinSides  — swap a join's probe/build    (column order/type mismatch)
//    * kReverseAggKeys — reverse composite GROUP BY    (key column type mismatch)
//    * kDropSort       — drop a Sort node              (POSITIONAL order mismatch)
//
//  Replay:  ./plan_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/join.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "plan/plan_mutants.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

// Diff an already-built (possibly mutant-lowered) engine tree against the plan
// oracle(s) — the reference always, DuckDB when staged. Ordered iff the plan's
// root is Sort (positional compare, WP-7).
DiffResult diff_tree_vs_plan_oracles(Operator& tree, const Plan& p) {
    const bool ordered = p.kind() == PlanKind::Sort;
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_plan_reference(p);
    DiffResult d = compare_result_sets(engine, ref, ordered);
    if (d.equal && duckdb_available()) {
        try {
            d = compare_result_sets(engine, run_plan_duckdb(p), ordered);
        } catch (const DuckDBError&) { /* divergence backstop; ignore */ }
    }
    return d;
}

}  // namespace

TEST_CASE("WP-8 mutation: dropped Filter in a deep pipeline is flagged") {
    // probe key 150 is filtered out (< 100); under LEFT join dropping the filter
    // re-introduces it as an extra group => row-count mismatch.
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 150, 3}));
    const Table probe(ps, std::move(pcols));

    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 2, 3}));
    bcols.push_back(i64_col({10, 20, 30}));
    const Table build(bs, std::move(bcols));

    const Plan p =
        scan(probe)
            .filter(lt(col(Type::I32, 0), lit(Scalar::i32(100))))
            .join(scan(build), {0}, {0}, JoinType::Left)
            .aggregate({0}, {AggSpec::count_star("n"), AggSpec::max(2, "mx")})
            .sort({SortKey{0, SortDir::Asc, NullOrder::Last},
                   SortKey{1, SortDir::Asc, NullOrder::Last},
                   SortKey{2, SortDir::Asc, NullOrder::Last}})
            .plan();

    CHECK(run_plan_vs_reference(p, 64).equal);  // correct lowering passes

    auto mutant = lower_mutant(p, LowerMutation::kDropFilter, 64);
    const DiffResult d = diff_tree_vs_plan_oracles(*mutant, p);
    CHECK_FALSE(d.equal);
    MESSAGE("dropped-filter mutant caught: " << d.message);
}

TEST_CASE("WP-8 mutation: swapped join probe/build sides is flagged") {
    // Distinct probe/build schemas => swapping reorders output column types.
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 3}));
    pcols.push_back(i64_col({10, 20, 30}));
    const Table probe(ps, std::move(pcols));

    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c2", Type::F64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 2, 3}));
    bcols.push_back(f64_col({1.5, 2.5, 3.5}));
    const Table build(bs, std::move(bcols));

    const Plan p =
        scan(probe).join(scan(build), {0}, {0}, JoinType::Inner).plan();

    CHECK(run_plan_vs_reference(p, 64).equal);

    auto mutant = lower_mutant(p, LowerMutation::kSwapJoinSides, 64);
    const DiffResult d = diff_tree_vs_plan_oracles(*mutant, p);
    CHECK_FALSE(d.equal);
    MESSAGE("swapped-join mutant caught: " << d.message);
}

TEST_CASE("WP-8 mutation: reversed composite GROUP BY keys is flagged") {
    // Composite key (i32, i64): reversing the key order reorders the output key
    // columns => column-type mismatch.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 2, 1, 2, 3}));
    cols.push_back(i64_col({7, 8, 7, 9, 7}));
    const Table t(s, std::move(cols));

    const Plan p = scan(t).aggregate({0, 1}, {AggSpec::count_star("n")}).plan();

    CHECK(run_plan_vs_reference(p, 64).equal);

    auto mutant = lower_mutant(p, LowerMutation::kReverseAggKeys, 64);
    const DiffResult d = diff_tree_vs_plan_oracles(*mutant, p);
    CHECK_FALSE(d.equal);
    MESSAGE("reversed-agg-keys mutant caught: " << d.message);
}

TEST_CASE("WP-8 mutation: dropped Sort is flagged under positional compare") {
    // Unsorted input: dropping the Sort leaves rows out of order => the POSITIONAL
    // comparator (root is Sort) bites where a canonicalizing compare would hide it.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({3, 1, 2, 5, 4}));
    const Table t(s, std::move(cols));

    const Plan p =
        scan(t).sort({SortKey{0, SortDir::Asc, NullOrder::Last}}).plan();
    REQUIRE(p.kind() == PlanKind::Sort);

    CHECK(run_plan_vs_reference(p, 64).equal);

    auto mutant = lower_mutant(p, LowerMutation::kDropSort, 64);
    const DiffResult d = diff_tree_vs_plan_oracles(*mutant, p);
    CHECK_FALSE(d.equal);
    MESSAGE("dropped-sort mutant caught: " << d.message);
}
