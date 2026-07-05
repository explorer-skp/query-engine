//  WP-13 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot fail
//  proves nothing"). We plant deliberately-broken Window operators
//  (tsx/window_mutants.{h,cpp}) — each a faithful copy of the real tumbling
//  group/scatter or sliding sort/sweep/emit loop reusing the SAME ops/join_internal.h
//  gather + ops/agg_internal.h finalizer + frozen Sort/HashTable, with exactly one
//  bad step — and SHOW the differential (independent reference + DuckDB window /
//  time-bucket when staged) CATCHES every mutant, while the REAL operator PASSES the
//  identical diff.
//
//  Planted mutants:
//    * kBucketEdge    — tumbling bucket edge shifted by one (bucket_id = (t+1)/W), so
//      rows near a boundary fall in the WRONG bucket.
//    * kFrameOffByOne — sliding frame reaches P+1 preceding rows, so every running
//      aggregate sees one extra row.
//
//  Replay:  ./window_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"
#include "tsx/window.h"
#include "tsx/window_mutants.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

OwnedColumn ts_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::TS, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}
OwnedColumn i64c(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

struct Case {
    Table input;
    tsx::WindowMode mode;
    std::vector<std::uint32_t> keys;
    std::uint32_t time;
    std::int64_t param;
    std::vector<AggSpec> aggs;
};

Plan case_plan(const Case& c) {
    std::vector<ColRef> ks;
    for (auto k : c.keys) ks.push_back(ColRef(static_cast<int>(k)));
    PlanBuilder b = scan(c.input);
    return (c.mode == tsx::WindowMode::Tumbling)
               ? b.window_tumbling(ks, ColRef(static_cast<int>(c.time)), c.param,
                                   c.aggs)
                     .plan()
               : b.window_sliding(ks, ColRef(static_cast<int>(c.time)), c.param,
                                  c.aggs)
                     .plan();
}

DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_window_reference(c.input, c.mode, c.keys, c.time,
                                               c.param, c.aggs);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            const ResultSet dk = run_plan_duckdb(case_plan(c));
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

DiffResult run_real(const Case& c, std::size_t bs) {
    auto sc = std::make_unique<Scan>(c.input, bs);
    tsx::Window w(std::move(sc), c.mode, c.keys, c.time, c.param, c.aggs);
    return diff_tree_vs_oracles(w, c);
}

DiffResult run_mutant(const Case& c, mutant::WindowMutation m, std::size_t bs) {
    auto sc = std::make_unique<Scan>(c.input, bs);
    mutant::Window w(std::move(sc), c.mode, c.keys, c.time, c.param, c.aggs, m);
    return diff_tree_vs_oracles(w, c);
}

// Tumbling, width 10, timestamps straddling bucket edges {9,10,19,20}: the correct
// bucketing is {0,10,10,20}; the kBucketEdge shift (t+1)/W gives {10,10,20,20}, so
// the grouping AND the emitted bucket_start diverge.
Case bucket_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::TS);  // timestamp
    s.fields.emplace_back("c1", Type::I64);  // payload
    std::vector<OwnedColumn> cols;
    cols.push_back(ts_col({9, 10, 19, 20}));
    cols.push_back(i64c({100, 200, 300, 400}));
    return {Table(s, std::move(cols)), tsx::WindowMode::Tumbling, {}, 0, 10,
            {AggSpec::count_star("n"), AggSpec::sum(1, "s")}};
}

// Sliding, P=1, one partition of 3 time-ascending rows: the correct frame at the
// last row is the 2 most-recent rows; kFrameOffByOne reaches a 3rd (P+1) row, so the
// running SUM/COUNT/MIN/MAX diverge.
Case frame_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);  // key
    s.fields.emplace_back("c1", Type::TS);   // timestamp
    s.fields.emplace_back("c2", Type::I64);  // payload
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1}));
    cols.push_back(ts_col({0, 10, 20}));
    cols.push_back(i64c({1, 2, 3}));
    return {Table(s, std::move(cols)), tsx::WindowMode::Sliding, {0}, 1, 1,
            {AggSpec::count_star("n"), AggSpec::sum(2, "s"),
             AggSpec::min(2, "mn"), AggSpec::max(2, "mx")}};
}

}  // namespace

TEST_CASE("MUTATION: the REAL window operator passes the differential") {
    for (std::size_t bs : {64u, 2048u}) {
        CHECK(run_real(bucket_case(), bs).equal);
        CHECK(run_real(frame_case(), bs).equal);
    }
}

TEST_CASE("MUTATION: kBucketEdge (tumbling bucket off by one) is CAUGHT") {
    const DiffResult d =
        run_mutant(bucket_case(), mutant::WindowMutation::kBucketEdge, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kBucketEdge caught: " << d.message);
}

TEST_CASE("MUTATION: kFrameOffByOne (sliding frame P+1) is CAUGHT") {
    const DiffResult d =
        run_mutant(frame_case(), mutant::WindowMutation::kFrameOffByOne, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kFrameOffByOne caught: " << d.message);
}
