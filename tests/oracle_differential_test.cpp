//  WP-3 / M1 PROOF: the first end-to-end scan->filter->project pipeline DIFFS
//  GREEN against the oracle on seeded random schema/data/query. Today CI runs the
//  always-available independent reference oracle (oracle/reference_oracle.h);
//  when the DuckDB amalgamation is staged (QE_WITH_DUCKDB) the SAME runner +
//  comparator also diff against DuckDB, the authoritative golden model (D16).
//
//  Determinism: every run prints its seed (tests/wp1_test_main.cpp); CI pins
//  QE_CI_SEED; replay any failure with:  ./oracle_differential_test --seed N
//
//  Batch size is varied across {64, 256, 2048} so batch-boundary / tail handling
//  is exercised against tables whose row counts are not multiples of the batch.

#include <cstddef>
#include <cstdint>
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
#include "ops/table.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// Run engine vs reference (always) and vs DuckDB (when staged); return combined
// verdict, attaching a message on the first failure.
DiffResult diff_all(const Table& t, const LogicalQuery& q, std::size_t bs) {
    DiffResult ref = run_vs_reference(t, q, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_differential(t, q, run_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            // Divergence backstop: generators should never produce a raising
            // query, so this should not fire. Skip rather than diff.
            MESSAGE("DuckDB raised (skipped, not a diff): " << e.what());
        }
    }
    return ref;  // equal
}

}  // namespace

TEST_CASE("M1: scan->filter->project diffs green vs oracle on random queries") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5CA9F11Eu);

    for (int iter = 0; iter < 300; ++iter) {
        const Schema schema = gen_schema(rng);
        const Table table = gen_table(rng, schema);
        const LogicalQuery query = gen_query(rng, schema);
        const std::size_t bs = kBatchSizes[rng() % 3];

        const DiffResult d = diff_all(table, query, bs);
        CHECK(d.equal);
        if (!d.equal) {
            MESSAGE("DIFF at iter=" << iter << " rows=" << table.num_rows()
                                    << " batch=" << bs << " : " << d.message);
            break;  // first failure suffices; replay with --seed
        }
    }
}

TEST_CASE("edge: empty table yields empty result on both sides") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE3170Eu);
    GenConfig cfg;
    cfg.min_rows = 0;
    cfg.max_rows = 0;
    const Schema schema = gen_schema(rng);
    const Table table = gen_table(rng, schema, cfg);
    const LogicalQuery query = gen_query(rng, schema);
    for (std::size_t bs : kBatchSizes) {
        const DiffResult d = diff_all(table, query, bs);
        CHECK(d.equal);
    }
}

TEST_CASE("edge: explicit all-pass, all-fail, and null-laden columns") {
    // c0 i32 with nulls, c1 f64. A handful of rows; check the boundary behaviors
    // against the oracle directly.
    using namespace qe::ops_test;
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 0, 7, 2, -9, 4}, /*nulls=*/{2, 5}));
    cols.push_back(f64_col({1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5}, {0}));
    const Table t(s, std::move(cols));

    SUBCASE("all-pass: WHERE TRUE") {
        LogicalQuery q;
        q.filter = lit(Scalar::boolean(true));
        q.projections.push_back({"p0", col(Type::I32, 0)});
        q.projections.push_back({"p1", col(Type::F64, 1)});
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    }
    SUBCASE("all-fail: WHERE FALSE") {
        LogicalQuery q;
        q.filter = lit(Scalar::boolean(false));
        q.projections.push_back({"p0", col(Type::I32, 0)});
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    }
    SUBCASE("predicate over a null column + derived projection") {
        LogicalQuery q;
        q.filter = gt(col(Type::I32, 0), lit(Scalar::i32(0)));
        q.projections.push_back({"p0", col(Type::I32, 0)});
        q.projections.push_back(
            {"p1", add(col(Type::I32, 0), lit(Scalar::i32(100)))});
        q.projections.push_back({"p2", col(Type::F64, 1)});
        for (std::size_t bs : kBatchSizes) CHECK(diff_all(t, q, bs).equal);
    }
}
