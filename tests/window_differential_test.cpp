//  WP-13: the windowed / time-bucketed aggregation differential. ONE Plan
//  (scan(input).window_tumbling(...) or .window_sliding(...)) drives BOTH backends:
//  it lowers to the tsx::Window operator and renders to DuckDB (a GROUP BY over the
//  integer time bucket for TUMBLING; a window function with ROWS BETWEEN P PRECEDING
//  AND CURRENT ROW for SLIDING). Each generated tick-like case diffs GREEN vs the
//  independent reference (always) AND, when staged (QE_WITH_DUCKDB), the AUTHORITATIVE
//  DuckDB diff. Window output is UNORDERED (no ORDER BY wraps it), so the comparator
//  canonicalizes (D12); float aggregates use the D11 tolerance.
//
//  Coverage: tumbling & sliding; 0 / 1 / 2 partition keys; bucket widths & frame
//  sizes from a small set; aggregate subsets (COUNT(*)/COUNT/SUM/MIN/MAX/AVG); NULL
//  partition keys (kEqual: group together); all-null / empty frame slices; row counts
//  spanning several 2048 batches incl. a tail; empty input. Batch size is swept across
//  {64,256,2048} so the sort/cursor straddle batch boundaries.
//
//  GENERATION CONSTRAINTS (see oracle/generators.cpp): timestamps are >= 0 (tumbling
//  integer bucketing is divergence-free vs DuckDB) AND globally distinct (sliding
//  ORDER BY t is a TOTAL order per partition => deterministic running values). Seed
//  printed (tests/wp1_test_main.cpp). Replay:  ./window_differential_test --seed N

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"
#include "tsx/window.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// Build the window Plan from an input table + spec (the SINGLE source of truth
// feeding both backends). `input` must outlive the returned Plan (Scan borrows).
Plan window_plan(const Table& input, tsx::WindowMode mode,
                 const std::vector<std::uint32_t>& keys, std::uint32_t time,
                 std::int64_t param, const std::vector<AggSpec>& aggs) {
    std::vector<ColRef> ks;
    for (auto k : keys) ks.push_back(ColRef(static_cast<int>(k)));
    PlanBuilder b = scan(input);
    return (mode == tsx::WindowMode::Tumbling)
               ? b.window_tumbling(ks, ColRef(static_cast<int>(time)), param, aggs)
                     .plan()
               : b.window_sliding(ks, ColRef(static_cast<int>(time)), param, aggs)
                     .plan();
}

// Diff a Plan vs the independent reference (always) and, when staged, DuckDB.
DiffResult diff_one(const Plan& p, std::size_t bs) {
    DiffResult ref = run_plan_vs_reference(p, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_plan_differential(p, run_plan_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            MESSAGE("DuckDB raised (skipped, not a diff): " << e.what());
        }
    }
    return ref;
}

bool diff_all_batches(const Plan& p) {
    for (std::size_t bs : kBatchSizes)
        if (!diff_one(p, bs).equal) {
            const DiffResult d = diff_one(p, bs);
            MESSAGE("DIFF at batch=" << bs << ": " << d.message << "\nplan:\n"
                                     << p.to_string());
            return false;
        }
    return true;
}

}  // namespace

TEST_CASE("WP-13: random windowed aggregations diff green vs reference + DuckDB") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x317D0Fu);
    for (int iter = 0; iter < 200; ++iter) {
        WindowCase c = gen_window_case(rng);
        const Plan p =
            window_plan(c.input, c.mode, c.keys, c.time, c.param, c.aggs);
        CHECK(diff_all_batches(p));
    }
}

// ---- explicit edge shapes ----------------------------------------------------

namespace {

// A dense I64 column from values (ops_test_util has i32/f64 but not i64).
OwnedColumn i64_col_local(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

// input: key c0 (i32), timestamp c1 (TS), payload c2 (i64), payload c3 (f64).
Table make_input() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::TS);
    s.fields.emplace_back("c2", Type::I64);
    s.fields.emplace_back("c3", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1, 2, 2, 1}));
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 6);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        const std::int64_t v[] = {0, 5, 12, 3, 21, 30};  // >=0, per-key distinct
        for (int i = 0; i < 6; ++i) d[i] = v[i];
        cols.push_back(std::move(t));
    }
    {
        OwnedColumn p = OwnedColumn::make(Type::I64, 6);
        auto* d = reinterpret_cast<std::int64_t*>(p.mutable_data());
        const std::int64_t v[] = {10, 20, 30, 40, 50, 60};
        for (int i = 0; i < 6; ++i) d[i] = v[i];
        p.set_null(2);  // exercise NULL aggregation
        cols.push_back(std::move(p));
    }
    cols.push_back(f64_col({1.5, 2.5, 3.5, 4.5, 5.5, 6.5}));
    return Table(s, std::move(cols));
}

const std::vector<std::uint32_t> kNoKeys{};
const std::vector<std::uint32_t> kKey0{0};

std::vector<AggSpec> all_aggs() {
    return {AggSpec::count_star("n"),  AggSpec::count(2, "cnt2"),
            AggSpec::sum(2, "sum2"),   AggSpec::min(2, "min2"),
            AggSpec::max(2, "max2"),   AggSpec::avg(3, "avg3")};
}

}  // namespace

TEST_CASE("WP-13 edge: tumbling, single key, width 10 (occupied buckets)") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, kKey0, 1, 10,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: tumbling, zero keys (global time buckets)") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, kNoKeys, 1, 8,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: tumbling, width 1 (one bucket per distinct timestamp)") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, kKey0, 1, 1,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: sliding, single key, P=2 (frame boundary)") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, kKey0, 1, 2,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: sliding, P=0 (current row only)") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, kKey0, 1, 0,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: sliding, zero keys (single global frame), wide P") {
    const Table in = make_input();
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, kNoKeys, 1, 100,
                                       all_aggs())));
}

TEST_CASE("WP-13 edge: NULL partition keys group together (kEqual)") {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::TS);
    s.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 7, 7}, /*nulls=*/{2, 3}));  // two NULL-key rows
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 4);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        const std::int64_t v[] = {0, 10, 5, 20};
        for (int i = 0; i < 4; ++i) d[i] = v[i];
        cols.push_back(std::move(t));
    }
    cols.push_back(i64_col_local({100, 200, 300, 400}));
    const Table in(s, std::move(cols));
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, kKey0, 1, 10,
                                       {AggSpec::count_star("n"),
                                        AggSpec::sum(2, "s")})));
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, kKey0, 1, 1,
                                       {AggSpec::count_star("n"),
                                        AggSpec::sum(2, "s")})));
}

TEST_CASE("WP-13 edge: empty input (tumbling => 0 rows; sliding => 0 rows)") {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({}));
    cols.push_back(OwnedColumn::make(Type::TS, 0));
    const Table in(s, std::move(cols));
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, kKey0, 1, 10,
                                       {AggSpec::count_star("n")})));
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, kKey0, 1, 3,
                                       {AggSpec::count_star("n")})));
}

TEST_CASE("WP-13 edge: I32 timestamp, composite key tumbling") {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I32);
    s.fields.emplace_back("c2", Type::I32);  // timestamp (I32)
    s.fields.emplace_back("c3", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1, 2}));
    cols.push_back(i32_col({7, 7, 9, 7}));
    cols.push_back(i32_col({0, 11, 4, 25}));  // >=0
    cols.push_back(i64_col_local({100, 200, 300, 400}));
    const Table in(s, std::move(cols));
    const std::vector<std::uint32_t> k01{0, 1};
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Tumbling, k01, 2, 10,
                                       {AggSpec::count_star("n"),
                                        AggSpec::sum(3, "s"),
                                        AggSpec::min(2, "mt")})));
    CHECK(diff_all_batches(window_plan(in, tsx::WindowMode::Sliding, k01, 2, 1,
                                       {AggSpec::count_star("n"),
                                        AggSpec::max(3, "mx")})));
}
