//  WP-5 / M2 PROOF: hash aggregation / GROUP BY diffs GREEN against the oracle on
//  seeded random schema/data/group-by queries, plus targeted edge cases. The
//  always-available independent reference oracle runs every case; when the DuckDB
//  amalgamation is staged (QE_WITH_DUCKDB) the SAME runner + comparator also diff
//  against DuckDB, the AUTHORITATIVE golden model (D16). GROUP BY has no inherent
//  row order, so the comparator canonicalizes (sorts both sides on all output
//  columns) before diffing (D12); float SUM/AVG compare within the D11 epsilon.
//
//  Determinism: every run prints its seed (tests/wp1_test_main.cpp). Replay:
//      ./aggregate_differential_test --seed N
//  Batch size is varied across {64,256,2048} so group-build across batch
//  boundaries / the output-batch tail (>2048 groups) are exercised.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::oracle;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

DiffResult diff_all(const Table& t, const LogicalQuery& q, std::size_t bs) {
    DiffResult ref = run_vs_reference(t, q, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_differential(t, q, run_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            FAIL("DuckDB raised on a grammar-safe generated case -- renderer/oracle regression, not a divergence: " << std::string(e.what()));
        }
    }
    return ref;
}

}  // namespace

TEST_CASE("M2: GROUP BY diffs green vs oracle on random queries") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6B0179u);

    for (int iter = 0; iter < 200; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const LogicalQuery query = gen_group_by_query(rng, schema);
        const std::size_t bs = kBatchSizes[rng() % 3];

        const DiffResult d = diff_all(table, query, bs);
        CHECK(d.equal);
        if (!d.equal) {
            MESSAGE("DIFF at iter=" << iter << " rows=" << table.num_rows()
                                    << " keys=" << query.group_by->keys.size()
                                    << " aggs=" << query.group_by->aggs.size()
                                    << " batch=" << bs << " : " << d.message);
            break;  // first failure suffices; replay with --seed
        }
    }
}

namespace {

using namespace qe::ops_test;

// c0 = i32 group key (with nulls), c1 = i32 value (with nulls), c2 = f64 value.
Table edge_table() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I32);
    s.fields.emplace_back("c2", Type::F64);
    std::vector<std::int32_t> k = {1, 1, 2, 2, 3, 1, 2, 3, 0, 0};
    std::vector<std::int32_t> v = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> f = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5};
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(k, /*nulls=*/{8, 9}));  // last two rows: NULL key group
    cols.push_back(i32_col(v, /*nulls=*/{0}));
    cols.push_back(f64_col(f, {2}));
    return Table(s, std::move(cols));
}

std::vector<AggSpec> full_aggs() {
    std::vector<AggSpec> a;
    a.push_back(AggSpec::count_star("cs"));
    a.push_back(AggSpec::count(1, "cnt1"));
    a.push_back(AggSpec::sum(1, "sum1"));
    a.push_back(AggSpec::min(1, "min1"));
    a.push_back(AggSpec::max(1, "max1"));
    a.push_back(AggSpec::avg(1, "avg1"));
    a.push_back(AggSpec::sum(2, "sum2"));
    a.push_back(AggSpec::avg(2, "avg2"));
    return a;
}

}  // namespace

TEST_CASE("edge: single key (incl. a NULL-key group)") {
    const Table t = edge_table();
    LogicalQuery q;
    q.group_by = GroupBy{{0}, full_aggs()};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: composite key (i32, f64)") {
    const Table t = edge_table();
    LogicalQuery q;
    q.group_by = GroupBy{{0, 2}, {AggSpec::count_star("cs"), AggSpec::sum(1, "s")}};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: global aggregate (zero keys)") {
    const Table t = edge_table();
    LogicalQuery q;
    q.group_by = GroupBy{{}, full_aggs()};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: all-null value group => SUM/MIN/MAX/AVG NULL, COUNT(col) 0") {
    // Group key 1 has BOTH its c1 values NULL; expect SUM/MIN/MAX/AVG NULL there.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I32);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 2, 2}));
    cols.push_back(i32_col({0, 0, 5, 6}, /*nulls=*/{0, 1}));  // group 1 all null
    const Table t(s, std::move(cols));
    LogicalQuery q;
    q.group_by = GroupBy{{0},
                         {AggSpec::count_star("cs"), AggSpec::count(1, "c1"),
                          AggSpec::sum(1, "s1"), AggSpec::min(1, "mn"),
                          AggSpec::max(1, "mx"), AggSpec::avg(1, "av")}};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: key skew (few distinct keys, many rows, >2048 -> tail batch)") {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<std::int32_t> k(6000);
    std::vector<std::int64_t> v(6000);
    for (int i = 0; i < 6000; ++i) {
        k[i] = i % 5;                 // 5 hot groups
        v[i] = (i % 100) - 50;        // bounded
    }
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(k));
    // i64 value column built via the ht_test helper would need extra include;
    // reuse i32 path by widening into an OwnedColumn directly.
    OwnedColumn c1 = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c1.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    cols.push_back(std::move(c1));
    const Table t(s, std::move(cols));

    LogicalQuery q;  // grouping by c1 (6000 distinct-ish) crosses the 2048 tail
    q.group_by = GroupBy{{1}, {AggSpec::count_star("cs"), AggSpec::sum(0, "s")}};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: empty input — global => 1 row, keyed => 0 rows") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE3170Eu);
    GenConfig cfg;
    cfg.min_rows = 0;
    cfg.max_rows = 0;
    const Schema schema = gen_schema(rng);
    const Table table = gen_table(rng, schema, cfg);

    SUBCASE("global over empty") {
        LogicalQuery q;
        q.group_by = GroupBy{{}, {AggSpec::count_star("cs"), AggSpec::count(0, "c0"),
                                  AggSpec::sum(0, "s0"), AggSpec::min(0, "m0")}};
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(table, q, bs).equal);
    }
    SUBCASE("keyed over empty") {
        LogicalQuery q;
        q.group_by = GroupBy{{0}, {AggSpec::count_star("cs"), AggSpec::sum(0, "s0")}};
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(table, q, bs).equal);
    }
}

TEST_CASE("edge: NaN values in F64 MIN/MAX/SUM/AVG (NaN-greatest total order)") {
    // Audit C2: DuckDB orders NaN greater than every other double, so
    //   MIN(group) is NaN only when every non-NULL input is NaN,
    //   MAX(group) is NaN as soon as any non-NULL input is NaN,
    //   SUM/AVG poison to NaN (IEEE), which float_eq treats as equal-to-NaN.
    // Pre-fix, raw std::min/max silently dropped NaNs (an all-NaN group emitted
    // the +/-inf identity) and the vector kernel was ISA-dependent.
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    // key 1: {NaN, 1.5}   -> MIN 1.5,  MAX NaN
    // key 2: {NaN, NaN}   -> MIN NaN,  MAX NaN   (the pre-fix +inf bug case)
    // key 3: {2.5, 3.5}   -> MIN 2.5,  MAX 3.5   (NaN-free control)
    // key 4: {NaN, NULL}  -> MIN NaN,  MAX NaN   (NULL ignored, not folded)
    std::vector<std::int32_t> k = {1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<double> f = {qnan, 1.5, qnan, qnan, 2.5, 3.5, qnan, 0.0};
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(k));
    cols.push_back(f64_col(f, /*nulls=*/{7}));
    const Table t(s, std::move(cols));

    std::vector<AggSpec> aggs;
    aggs.push_back(AggSpec::min(1, "mn"));
    aggs.push_back(AggSpec::max(1, "mx"));
    aggs.push_back(AggSpec::sum(1, "sm"));
    aggs.push_back(AggSpec::avg(1, "av"));
    aggs.push_back(AggSpec::count(1, "ct"));

    SUBCASE("grouped") {
        LogicalQuery q;
        q.group_by = GroupBy{{0}, aggs};
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    }
    SUBCASE("global (kernel path)") {
        LogicalQuery q;
        q.group_by = GroupBy{{}, aggs};
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    }
}
