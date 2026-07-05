//  WP-7 PROOF: ORDER BY diffs GREEN vs the oracle under POSITIONAL compare (D12).
//  CI runs the always-available independent reference oracle; when the DuckDB
//  amalgamation is staged (QE_WITH_DUCKDB) the SAME runner + comparator also diff
//  against DuckDB, the authoritative golden model (D16). Because the query carries
//  an explicit ORDER BY, run_differential compares ROW-BY-ROW in emitted order
//  (ORDERED mode) — a wrong sort is visible, which a canonicalizing compare would
//  hide.
//
//  Coverage: seeded random schema/data/query (single & multi key, ASC/DESC mixes,
//  NULLS FIRST/LAST, all types, ties from bounded magnitudes, multi-batch emit),
//  plus explicit edge cases (empty input, single-key DESC, NULLS placement,
//  >2048 rows). The generator appends every remaining column as a deterministic
//  tiebreak so the order is TOTAL and positional compare is unambiguous (see
//  oracle/generators.h and the WP report on float/NaN determinism).
//
//  Determinism: seed printed (tests/wp1_test_main.cpp); replay with
//      ./sort_differential_test --seed N

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// engine vs reference (always) and vs DuckDB (when staged); both POSITIONAL since
// the query has ORDER BY (run_differential passes ordered = q.has_order_by()).
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

TEST_CASE("WP-7: ORDER BY diffs green vs oracle on random queries (positional)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x507AB1Eu);
    for (int iter = 0; iter < 300; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const LogicalQuery query = gen_order_by_query(rng, schema);
        const std::size_t bs = kBatchSizes[rng() % 3];

        const DiffResult d = diff_all(table, query, bs);
        CHECK(d.equal);
        if (!d.equal) {
            MESSAGE("DIFF at iter=" << iter << " rows=" << table.num_rows()
                                    << " batch=" << bs << " : " << d.message);
            break;
        }
    }
}

TEST_CASE("edge: empty input sorts to empty result on both sides") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE3170Eu);
    GenConfig cfg;
    cfg.min_rows = 0;
    cfg.max_rows = 0;
    const Schema schema = gen_schema(rng);
    const Table table = gen_table(rng, schema, cfg);
    const LogicalQuery query = gen_order_by_query(rng, schema);
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(table, query, bs).equal);
}

TEST_CASE("edge: >2048 rows exercises multi-batch sorted emit") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xB1607u);
    GenConfig cfg;
    cfg.min_rows = 5000;
    cfg.max_rows = 6000;
    const Schema schema = gen_schema(rng);
    const Table table = gen_table(rng, schema, cfg);
    const LogicalQuery query = gen_order_by_query(rng, schema);
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(table, query, bs).equal);
}

TEST_CASE("edge: explicit single-key DESC + NULLS placement, all directions") {
    using namespace qe::ops_test;
    // c0 i32 with nulls (UNIQUE non-null values so single-key order is total),
    // c1 f64. Project both, ORDER BY c0 only — a genuine partial sort with a
    // deterministic order because the key values are distinct.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 11, 7, 2, -9, 4, 100}, /*nulls=*/{2, 5}));
    cols.push_back(f64_col({1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, {0}));
    const Table t(s, std::move(cols));

    auto run = [&](SortDir dir, NullOrder nulls) {
        LogicalQuery q;
        q.projections.push_back({"p0", col(Type::I32, 0)});
        q.projections.push_back({"p1", col(Type::F64, 1)});
        q.order_by = std::vector<SortKey>{SortKey{0, dir, nulls}};
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    };
    run(SortDir::Asc, NullOrder::First);
    run(SortDir::Asc, NullOrder::Last);
    run(SortDir::Desc, NullOrder::First);
    run(SortDir::Desc, NullOrder::Last);
}

TEST_CASE("edge: ORDER BY on a GROUP BY result (keys then aggregate)") {
    // A grouped query, then ORDER BY the aggregate DESC + the key ASC. Output is
    // (c0 key, sum). Keys+aggregate together make the order total.
    using namespace qe::ops_test;
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 2, 1, 3, 2, 1, 3, 2}));
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, 8);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        const std::int64_t v[] = {10, 20, 30, 5, 7, 1, 9, 3};
        for (int i = 0; i < 8; ++i) d[i] = v[i];
        cols.push_back(std::move(c));
    }
    const Table t(s, std::move(cols));
    LogicalQuery q;
    q.group_by = GroupBy{{0}, {AggSpec::sum(1, "s1")}};
    // Output cols: 0 = c0 (key), 1 = s1 (sum). ORDER BY sum DESC, key ASC.
    q.order_by = std::vector<SortKey>{
        SortKey{1, SortDir::Desc, NullOrder::Last},
        SortKey{0, SortDir::Asc, NullOrder::Last}};
    for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
}

TEST_CASE("edge: NaN in an F64 sort key (NaN-greatest, all directions)") {
    // Audit C2: DuckDB orders NaN greater than every non-NaN double; the engine
    // comparator now implements the same TOTAL order (pre-fix, NaN keys made the
    // comparator non-strict-weak — std::stable_sort UB). The unique I64 column
    // is appended as a tiebreaker so the positional compare is deterministic
    // (all NaNs tie with each other).
    using namespace qe::ops_test;
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Schema s;
    s.fields.emplace_back("c0", Type::F64);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<double> f = {qnan, 2.0, -inf, qnan, 0.5, inf, -3.5, qnan, 7.25};
    std::vector<OwnedColumn> cols;
    cols.push_back(f64_col(f, /*nulls=*/{4}));  // one NULL among the NaNs
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, f.size());
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        for (std::size_t i = 0; i < f.size(); ++i) d[i] = static_cast<std::int64_t>(i);
        cols.push_back(std::move(c));
    }
    const Table t(s, std::move(cols));
    for (auto dir : {SortDir::Asc, SortDir::Desc})
        for (auto no : {NullOrder::First, NullOrder::Last}) {
            LogicalQuery q;
            q.projections.push_back({"p0", qe::expr::col(Type::F64, 0)});
            q.projections.push_back({"p1", qe::expr::col(Type::I64, 1)});
            q.order_by = std::vector<SortKey>{
                SortKey{0, dir, no},
                SortKey{1, SortDir::Asc, NullOrder::Last}};  // unique tiebreaker
            for (std::size_t bs : kBatchSizes) {
                const DiffResult d = diff_all(t, q, bs);
                CHECK_MESSAGE(d.equal, "dir=" << std::string(dir == SortDir::Asc ? "ASC" : "DESC")
                                              << " nulls=" << std::string(no == NullOrder::First ? "FIRST" : "LAST")
                                              << " bs=" << bs << " : " << d.message);
            }
        }
}
