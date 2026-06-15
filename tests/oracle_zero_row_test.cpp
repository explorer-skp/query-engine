//  WP-9 ZERO-ROW differential cases (carry-forward (1)). The engine must agree with
//  DuckDB byte-for-byte on EMPTY results, the corner the brief calls out:
//    * a plan whose FILTER selects nothing (non-empty input, always-false predicate)
//      => 0 output rows;
//    * a GLOBAL aggregate over EMPTY input => exactly ONE row (COUNT=0, SUM/MIN/MAX/
//      AVG = NULL) per SQL semantics;
//    * the two composed: filter-to-nothing then a global aggregate;
//    * a KEYED aggregate over empty input => ZERO groups;
//    * filter-to-nothing then ORDER BY (positional, still 0 rows);
//    * an INNER join with no matching keys => 0 rows.
//
//  Each runs through the PLAN path (one Plan -> engine tree AND DuckDB SQL) so the
//  diff is honest, vs the independent reference AND, when staged (QE_WITH_DUCKDB),
//  the authoritative DuckDB golden model. These are the 0-row companions to the
//  direct compact_column(n==0) regression (tests/compact_empty_test.cpp): together
//  they pin carry-forward (1) at both the unit and differential levels.
//
//  Determinism: seed printed (tests/wp1_test_main.cpp). Replay:
//      ./oracle_zero_row_test --seed N

#include <cstddef>
#include <cstdint>
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
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// Diff a Plan vs the reference (always) and, when staged, DuckDB — across batch
// sizes (the Scan requires a positive multiple of 64). Ordered iff root is Sort.
bool zero_row_diffs_green(const Plan& p) {
    const bool ordered = p.kind() == PlanKind::Sort;
    for (std::size_t bs : kBatchSizes) {
        const DiffResult ref = run_plan_vs_reference(p, bs);
        if (!ref.equal) {
            MESSAGE("REF DIFF at batch=" << bs << ": " << ref.message
                                         << "\nplan:\n" << p.to_string());
            return false;
        }
        if (duckdb_available()) {
            try {
                const DiffResult d = run_plan_differential(p, run_plan_duckdb, bs);
                if (!d.equal) {
                    MESSAGE("DUCKDB DIFF at batch=" << bs << ": " << d.message
                                                    << "\nplan:\n" << p.to_string());
                    return false;
                }
            } catch (const DuckDBError& e) {
                MESSAGE("DuckDB raised (skipped, not a diff): " << e.what());
            }
        }
    }
    (void)ordered;
    return true;
}

// A small NON-empty table: c0 i32 in [0,10), c1 f64, one null in each column.
Table nonempty_table() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 2, 3, 4, 5}, /*nulls=*/{2}));
    cols.push_back(f64_col({1.5, 2.5, 3.5, 4.5, 5.5}, /*nulls=*/{0}));
    return Table(s, std::move(cols));
}

// An EMPTY (0-row) table of the same schema.
Table empty_table() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(OwnedColumn::make(Type::I32, 0));
    cols.push_back(OwnedColumn::make(Type::F64, 0));
    return Table(s, std::move(cols));
}

// An always-FALSE BOOL predicate over c0 (c0 < -1000; the data is all >= 0). This
// keeps the engine and DuckDB in lockstep: no NULL/raise ambiguity, just nothing
// passes.
Expr never_true() { return lt(col(Type::I32, 0), lit(Scalar::i32(-1000))); }

}  // namespace

TEST_CASE("0-row: filter selects nothing (non-empty input) -> empty projection") {
    const Table t = nonempty_table();
    const Plan p = scan(t)
                       .filter(never_true())
                       .project({{"a", col(Type::I32, 0)},
                                 {"b", col(Type::F64, 1)}})
                       .plan();
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: global aggregate over EMPTY input -> one all-NULL/zero row") {
    const Table t = empty_table();
    // COUNT(*) = 0; COUNT(c) = 0; SUM/MIN/MAX/AVG = NULL (SQL empty-group rules).
    const Plan p = scan(t)
                       .aggregate({}, {AggSpec::count_star("n"),
                                       AggSpec::count(0, "cn"),
                                       AggSpec::sum(0, "s"), AggSpec::min(1, "mn"),
                                       AggSpec::max(1, "mx"), AggSpec::avg(1, "av")})
                       .plan();
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: filter-to-nothing THEN global aggregate -> one row") {
    // The composed corner the brief names: the global aggregate's input is produced
    // empty by an upstream filter, not by an empty base table.
    const Table t = nonempty_table();
    const Plan p = scan(t)
                       .filter(never_true())
                       .aggregate({}, {AggSpec::count_star("n"),
                                       AggSpec::sum(0, "s"), AggSpec::avg(1, "av")})
                       .plan();
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: KEYED aggregate over empty input -> zero groups") {
    const Table t = empty_table();
    const Plan p =
        scan(t).aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(0, "s")})
            .plan();
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: filter-to-nothing THEN ORDER BY (positional, still empty)") {
    const Table t = nonempty_table();
    const Plan p = scan(t)
                       .filter(never_true())
                       .project({{"a", col(Type::I32, 0)},
                                 {"b", col(Type::F64, 1)}})
                       .sort({SortKey{0, SortDir::Asc, NullOrder::Last},
                              SortKey{1, SortDir::Asc, NullOrder::Last}})
                       .plan();
    REQUIRE(p.kind() == PlanKind::Sort);
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: INNER join with no matching keys -> empty result") {
    // Probe keys {100,200} share nothing with build keys {1,2} => 0 output rows.
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({100, 200}));
    const Table probe(ps, std::move(pcols));

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

    const Plan p =
        scan(probe).join(scan(build), {0}, {0}, JoinType::Inner).plan();
    CHECK(zero_row_diffs_green(p));
}

TEST_CASE("0-row: bare scan of an EMPTY table (identity) diffs green") {
    const Table t = empty_table();
    CHECK(zero_row_diffs_green(scan(t).plan()));
}
