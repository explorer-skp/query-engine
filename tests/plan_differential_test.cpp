//  WP-8 / M2-CLOSE PROOF: the plan + dataframe builder REPRODUCES every query
//  shape WP-3..WP-7 exercised — scan; scan->filter->project; group-by
//  (single/composite/global); inner & left join (single/composite key); ORDER BY
//  (multi-col, ASC/DESC, NULLS FIRST/LAST) — PLUS a DEEP composite pipeline no
//  single earlier WP exercised (scan->filter->join->aggregate->sort). Each diffs
//  GREEN through the PLAN path vs the independent plan reference AND, when staged
//  (QE_WITH_DUCKDB), the authoritative DuckDB plan diff. ONE Plan drives both: it
//  lowers to the engine tree and renders to the SQL DuckDB runs.
//
//  A plan ending in Sort diffs POSITIONALLY (ORDERED, D12); otherwise canonicalized
//  — run_plan_differential picks the mode from the root node kind.
//
//  Determinism: seed printed (tests/wp1_test_main.cpp). Replay:
//      ./plan_differential_test --seed N
//  Batch size is varied across {64,256,2048} so batch-boundary / tail handling and
//  the join fan-out tail are exercised against the lowered tree.

#include <cstddef>
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
#include "oracle/generators.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// Diff a Plan vs the independent plan reference (always) and, when staged, DuckDB.
DiffResult plan_diff_all(const Plan& p, std::size_t bs) {
    DiffResult ref = run_plan_vs_reference(p, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_plan_differential(p, run_plan_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            FAIL("DuckDB raised on a grammar-safe generated case -- renderer/oracle regression, not a divergence: " << std::string(e.what()));
        }
    }
    return ref;
}

bool diff_all_batches(const Plan& p) {
    for (std::size_t bs : kBatchSizes)
        if (!plan_diff_all(p, bs).equal) {
            const DiffResult d = plan_diff_all(p, bs);
            MESSAGE("DIFF at batch=" << bs << ": " << d.message
                                     << "\nplan:\n" << p.to_string());
            return false;
        }
    return true;
}

// Translate a single-table LogicalQuery into the equivalent Plan — mirrors
// oracle/logical_query.cpp::build_engine_pipeline, so lowering produces the SAME
// operator tree the WP-3..WP-7 tests build by hand.
Plan lq_to_plan(const Table& t, const LogicalQuery& q) {
    PlanBuilder b = scan(t);
    if (q.has_filter()) b = b.filter(q.filter);
    if (q.has_group_by()) {
        std::vector<ColRef> keys;
        for (auto k : q.group_by->keys) keys.push_back(ColRef(static_cast<int>(k)));
        b = b.aggregate(keys, q.group_by->aggs);
    } else {
        b = b.project(q.projections);
    }
    if (q.has_order_by()) b = b.sort(*q.order_by);
    return b.plan();
}

}  // namespace

// ---- random coverage: every WP-3..WP-7 shape, now through the PLAN path -------

TEST_CASE("WP-8: scan->filter->project plans diff green vs oracle (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x9180Au);
    for (int iter = 0; iter < 120; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_query(rng, schema));
        CHECK(diff_all_batches(p));
    }
}

TEST_CASE("WP-8: GROUP BY plans diff green vs oracle (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6B0B1Eu);
    for (int iter = 0; iter < 120; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_group_by_query(rng, schema));
        CHECK(diff_all_batches(p));
    }
}

TEST_CASE("WP-8: ORDER BY plans diff green vs oracle (random, positional)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x0D5A17u);
    for (int iter = 0; iter < 120; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_order_by_query(rng, schema));
        REQUIRE(p.kind() == PlanKind::Sort);  // positional compare engaged
        CHECK(diff_all_batches(p));
    }
}

TEST_CASE("WP-8: join plans (inner+left, single/composite) diff green (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x70112Du);
    for (int iter = 0; iter < 120; ++iter) {
        JoinCase jc = gen_join_case(rng);
        std::vector<ColRef> lk, rk;
        for (auto i : jc.query.probe_keys) lk.push_back(ColRef(static_cast<int>(i)));
        for (auto i : jc.query.build_keys) rk.push_back(ColRef(static_cast<int>(i)));
        const Plan p =
            scan(jc.probe).join(scan(jc.build), lk, rk, jc.query.type).plan();
        CHECK(diff_all_batches(p));
    }
}

// ---- the DEEP composite pipeline (the WP-8 differentiator) --------------------

TEST_CASE("WP-8: deep scan->filter->join->aggregate->sort diffs green (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xDEE9u);
    for (int iter = 0; iter < 80; ++iter) {
        PlanCase pc = gen_plan_case(rng);
        REQUIRE(pc.plan.kind() == PlanKind::Sort);
        CHECK(diff_all_batches(pc.plan));
    }
}

// ---- explicit edge shapes ----------------------------------------------------

TEST_CASE("WP-8 edge: bare scan diffs green (identity)") {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 0, 7}, /*nulls=*/{2}));
    cols.push_back(f64_col({1.5, 2.5, 3.5, 4.5}, {0}));
    const Table t(s, std::move(cols));
    CHECK(diff_all_batches(scan(t).plan()));
}

TEST_CASE("WP-8 edge: empty table through a filter->project plan") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE3170Eu);
    GenConfig cfg;
    cfg.min_rows = 0;
    cfg.max_rows = 0;
    const Schema schema = gen_schema(rng, cfg);
    const Table table = gen_table(rng, schema, cfg);
    const Plan p = lq_to_plan(table, gen_query(rng, schema));
    CHECK(diff_all_batches(p));
}

TEST_CASE("WP-8 edge: global aggregate (zero group keys)") {
    Schema s;
    s.fields.emplace_back("c0", Type::I64);
    std::vector<OwnedColumn> cols;
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, 5);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        const std::int64_t v[] = {10, 20, 30, 40, 50};
        for (int i = 0; i < 5; ++i) d[i] = v[i];
        cols.push_back(std::move(c));
    }
    const Table t(s, std::move(cols));
    const Plan p = scan(t)
                       .aggregate({}, {AggSpec::count_star("n"),
                                       AggSpec::sum(0, "s"), AggSpec::avg(0, "a"),
                                       AggSpec::min(0, "mn"), AggSpec::max(0, "mx")})
                       .plan();
    CHECK(diff_all_batches(p));
}

TEST_CASE("WP-8 edge: composite-key inner join, then group-by, then order-by") {
    // build: (c0 i32, c1 f64) key, c2 i64 payload.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::F64);
    bs.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2, 2}));
    bcols.push_back(f64_col({1.0, 2.0, 1.0, 2.0}));
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, 4);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        const std::int64_t v[] = {11, 12, 21, 22};
        for (int i = 0; i < 4; ++i) d[i] = v[i];
        bcols.push_back(std::move(c));
    }
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1, 2, 2}));
    pcols.push_back(f64_col({2.0, 9.0, 1.0, 2.0}));
    const Table probe(ps, std::move(pcols));

    // probe |x| build on (c0,c1), GROUP BY probe c0, COUNT(*)+MAX(build c2),
    // ORDER BY the key then the aggregates (total order).
    const Plan p =
        scan(probe)
            .join(scan(build), {0, 1}, {"c0", "c1"}, JoinType::Inner)
            .aggregate({0}, {AggSpec::count_star("n"), AggSpec::max(4, "mx")})
            .sort({SortBy{0}, SortBy{"n"}, SortBy{"mx"}})
            .plan();
    CHECK(diff_all_batches(p));
}

TEST_CASE("WP-8 edge: left join keeps unmatched probe rows (NULL build cols)") {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 2}));
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, 2);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        d[0] = 10; d[1] = 20;
        bcols.push_back(std::move(c));
    }
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 3, 2, 9}, /*nulls=*/{3}));  // 3,NULL unmatched
    const Table probe(ps, std::move(pcols));

    CHECK(diff_all_batches(
        scan(probe).join(scan(build), {0}, {0}, JoinType::Left).plan()));
}
