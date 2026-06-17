//  WP-10b DIFFERENTIAL — the optional parallel-execution path must produce the
//  SAME logical result as the single-thread plan path, under the oracle. For every
//  generated plan shape (the SAME generators the WP-8 plan differential uses —
//  gen_query, gen_group_by_query, gen_order_by_query, gen_join_case, gen_plan_case),
//  this runs the engine through the PARALLEL driver (exec/parallel.h) at several
//  THREAD counts {1,2,4,8} and several MORSEL sizes, and asserts:
//
//    parallel  ==  single-thread engine   (the IDENTICAL-TO-SINGLE-THREAD claim —
//                                           a merge bug that fooled DuckDB can't hide)
//    parallel  ==  independent reference oracle
//    parallel  ==  DuckDB                 (when the amalgamation is staged)
//
//  Canonicalized (D12) for unordered shapes; POSITIONAL for a top-level Sort. Varying
//  thread count AND morsel size makes group keys / join probes land in different
//  workers and morsels, so a partial-merge or morsel-range bug surfaces.
//
//  Determinism: seed printed (tests/wp1_test_main.cpp). Replay:
//      ./wp10b_parallel_differential_test --seed N

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "exec/parallel.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using qe::exec::ParallelConfig;
using qe::exec::run_plan_parallel;

namespace {

const unsigned kThreadCounts[] = {1, 2, 4, 8};
const std::size_t kMorselSizes[] = {64, 128, 512};

// Single-thread engine result (the identical-to claim's reference) for plan `p`.
ResultSet single_thread(const Plan& p, std::size_t bs) {
    std::unique_ptr<Operator> tree = p.lower(bs);
    return drain_operator(*tree);
}

// Diff the parallel driver at every {thread, morsel} against single-thread, the
// reference oracle, and DuckDB (when staged). Returns the first failing message.
bool parallel_matches_all(const Plan& p, std::string& why) {
    const bool ordered = p.kind() == PlanKind::Sort;
    const ResultSet st = single_thread(p, 2048);
    const ResultSet ref = run_plan_reference(p);
    // First, single-thread itself must agree with the reference (sanity).
    {
        const DiffResult d = compare_result_sets(st, ref, ordered);
        if (!d.equal) { why = "single-thread != reference: " + d.message; return false; }
    }
    ResultSet duck;
    bool have_duck = false;
    if (duckdb_available()) {
        try { duck = run_plan_duckdb(p); have_duck = true; }
        catch (const DuckDBError&) { have_duck = false; }
    }

    for (unsigned t : kThreadCounts) {
        for (std::size_t m : kMorselSizes) {
            ParallelConfig cfg;
            cfg.threads = t;
            cfg.morsel_rows = m;
            const ResultSet par = run_plan_parallel(p, cfg);

            DiffResult d = compare_result_sets(par, st, ordered);
            if (!d.equal) {
                why = "parallel(t=" + std::to_string(t) + ",m=" +
                      std::to_string(m) + ") != single-thread: " + d.message;
                return false;
            }
            d = compare_result_sets(par, ref, ordered);
            if (!d.equal) {
                why = "parallel(t=" + std::to_string(t) + ",m=" +
                      std::to_string(m) + ") != reference: " + d.message;
                return false;
            }
            if (have_duck) {
                d = compare_result_sets(par, duck, ordered);
                if (!d.equal) {
                    why = "parallel(t=" + std::to_string(t) + ",m=" +
                          std::to_string(m) + ") != DuckDB: " + d.message;
                    return false;
                }
            }
        }
    }
    return true;
}

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

TEST_CASE("WP-10b: scan->filter->project parallel == single-thread (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xB10B0u);
    for (int iter = 0; iter < 60; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_query(rng, schema));
        std::string why;
        CHECK_MESSAGE(parallel_matches_all(p, why), why << "\n" << p.to_string());
    }
}

TEST_CASE("WP-10b: GROUP BY parallel partial-agg merge == single-thread (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6B0Bu);
    for (int iter = 0; iter < 80; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_group_by_query(rng, schema));
        std::string why;
        CHECK_MESSAGE(parallel_matches_all(p, why), why << "\n" << p.to_string());
    }
}

TEST_CASE("WP-10b: ORDER BY parallel (upstream||, single-thread sort) == single") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x0D5Au);
    for (int iter = 0; iter < 60; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const Plan p = lq_to_plan(table, gen_order_by_query(rng, schema));
        REQUIRE(p.kind() == PlanKind::Sort);  // positional compare engaged
        std::string why;
        CHECK_MESSAGE(parallel_matches_all(p, why), why << "\n" << p.to_string());
    }
}

TEST_CASE("WP-10b: join probe parallelized (inner+left) == single-thread (random)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x701Du);
    for (int iter = 0; iter < 60; ++iter) {
        JoinCase jc = gen_join_case(rng);
        std::vector<ColRef> lk, rk;
        for (auto i : jc.query.probe_keys) lk.push_back(ColRef(static_cast<int>(i)));
        for (auto i : jc.query.build_keys) rk.push_back(ColRef(static_cast<int>(i)));
        const Plan p =
            scan(jc.probe).join(scan(jc.build), lk, rk, jc.query.type).plan();
        std::string why;
        CHECK_MESSAGE(parallel_matches_all(p, why), why << "\n" << p.to_string());
    }
}

TEST_CASE("WP-10b: deep scan->filter->join->aggregate->sort parallel == single") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xDEE9u);
    for (int iter = 0; iter < 50; ++iter) {
        PlanCase pc = gen_plan_case(rng);
        REQUIRE(pc.plan.kind() == PlanKind::Sort);
        std::string why;
        CHECK_MESSAGE(parallel_matches_all(pc.plan, why),
                      why << "\n" << pc.plan.to_string());
    }
}

TEST_CASE("WP-10b edge: empty table through parallel filter->project") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE317u);
    GenConfig cfg;
    cfg.min_rows = 0;
    cfg.max_rows = 0;
    const Schema schema = gen_schema(rng, cfg);
    const Table table = gen_table(rng, schema, cfg);
    const Plan p = lq_to_plan(table, gen_query(rng, schema));
    std::string why;
    CHECK_MESSAGE(parallel_matches_all(p, why), why);
}

TEST_CASE("WP-10b edge: global aggregate (zero keys) over empty input -> one row") {
    Schema s;
    s.fields.emplace_back("c0", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(OwnedColumn::make(Type::I64, 0));
    const Table t(s, std::move(cols));
    const Plan p = scan(t)
                       .aggregate({}, {AggSpec::count_star("n"),
                                       AggSpec::sum(0, "s"), AggSpec::avg(0, "a"),
                                       AggSpec::min(0, "mn"), AggSpec::max(0, "mx")})
                       .plan();
    std::string why;
    CHECK_MESSAGE(parallel_matches_all(p, why), why);
}
