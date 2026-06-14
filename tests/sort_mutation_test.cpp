//  WP-7 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken SORT operators
//  (ops/sort_mutants.{h,cpp}) — each a faithful copy of the real permutation/emit
//  loop with exactly one bad step — and SHOW the differential CATCHES every mutant
//  under POSITIONAL compare (D12), while the REAL operator PASSES the identical
//  diff. The positional comparator is the whole point: a canonicalizing compare
//  would re-sort both sides and HIDE a wrong row order, so a sort mutant is only
//  meaningful under ORDERED mode (verified here).
//
//  Planted mutants:
//    * kDescSortsAsc      — DESC is sorted ASC.
//    * kNullsFlipped      — NULLS FIRST/LAST is flipped.
//    * kRadixSignBug      — the radix encode omits the sign-bit flip (negatives
//      sort after non-negatives) — the classic radix bug.
//    * kUnstableTiebreak  — equal-key rows are emitted in REVERSED original order
//      (observable on a PARTIAL key vs the stable reference; DuckDB's tie order is
//      undefined, so this one is checked against the reference only).
//    * kEmitTailOffByOne  — the output drops the last sorted row.
//
//  Replay:  ./sort_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/sort_mutants.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

struct Case {
    Table table;
    LogicalQuery query;
};

// Project every column as-is, then ORDER BY `keys` — so the oracle's output schema
// matches the engine tree (Scan -> Sort over the raw table).
LogicalQuery make_query(const Schema& s, std::vector<SortKey> keys) {
    LogicalQuery q;
    for (std::uint32_t i = 0; i < s.fields.size(); ++i)
        q.projections.push_back(
            {"p" + std::to_string(i), col(s.fields[i].second, i)});
    q.order_by = std::move(keys);
    return q;
}

OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

DiffResult engine_vs_oracles(Operator& tree, const Case& c, bool also_duckdb) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_reference(c.table, c.query);
    DiffResult d = compare_result_sets(engine, ref, /*ordered=*/true);
    if (d.equal && also_duckdb && duckdb_available()) {
        try {
            const ResultSet dk = run_duckdb(c.table, c.query);
            d = compare_result_sets(engine, dk, /*ordered=*/true);
        } catch (const DuckDBError&) { /* divergence backstop; ignore */ }
    }
    return d;
}

DiffResult run_mutant(const Case& c, mutant::SortMutation m, bool also_duckdb,
                      std::size_t bs = 64) {
    auto scan = std::make_unique<Scan>(c.table, bs);
    mutant::Sort sort(std::move(scan), *c.query.order_by, m);
    return engine_vs_oracles(sort, c, also_duckdb);
}

// --- Cases ------------------------------------------------------------------

// Unique-valued integer keys (incl negatives) => single-key order is TOTAL.
Case desc_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 11, 7, 2, -9, 4, 100}));
    cols.push_back(i64_col({1, 2, 3, 4, 5, 6, 7, 8}));
    LogicalQuery q =
        make_query(s, {SortKey{0, SortDir::Desc, NullOrder::Last}});
    return Case{Table(s, std::move(cols)), std::move(q)};
}

// Two NULL rows distinguished by a unique second key, ordered NULLS FIRST — total.
Case nulls_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 11, 7, 2, -9}, /*nulls=*/{1, 4}));
    cols.push_back(i64_col({10, 20, 30, 40, 50, 60}));  // unique tiebreak
    LogicalQuery q = make_query(
        s, {SortKey{0, SortDir::Asc, NullOrder::First},
            SortKey{1, SortDir::Asc, NullOrder::Last}});
    return Case{Table(s, std::move(cols)), std::move(q)};
}

// Duplicate keys with distinct payload + a PARTIAL key (c0 only): stability is
// observable. Reference is stable; the mutant reverses ties.
Case tie_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 2, 2, 3, 3}));
    cols.push_back(i64_col({10, 20, 30, 40, 50, 60}));  // distinct payload
    LogicalQuery q =
        make_query(s, {SortKey{0, SortDir::Asc, NullOrder::Last}});
    return Case{Table(s, std::move(cols)), std::move(q)};
}

}  // namespace

TEST_CASE("MUTATION: the REAL sort passes the positional differential") {
    {
        const Case c = desc_case();
        for (std::size_t bs : {64u, 2048u})
            CHECK(run_vs_reference(c.table, c.query, bs).equal);
    }
    {
        const Case c = nulls_case();
        CHECK(run_vs_reference(c.table, c.query, 64).equal);
    }
    {
        const Case c = tie_case();
        CHECK(run_vs_reference(c.table, c.query, 64).equal);
    }
}

TEST_CASE("MUTATION: kDescSortsAsc (DESC sorted ASC) is CAUGHT") {
    const Case c = desc_case();
    const DiffResult d = run_mutant(c, mutant::SortMutation::kDescSortsAsc, true);
    CHECK_FALSE(d.equal);
    MESSAGE("kDescSortsAsc caught: " << d.message);
}

TEST_CASE("MUTATION: kNullsFlipped (NULLS FIRST/LAST flipped) is CAUGHT") {
    const Case c = nulls_case();
    const DiffResult d = run_mutant(c, mutant::SortMutation::kNullsFlipped, true);
    CHECK_FALSE(d.equal);
    MESSAGE("kNullsFlipped caught: " << d.message);
}

TEST_CASE("MUTATION: kRadixSignBug (sign bit dropped in radix) is CAUGHT") {
    const Case c = desc_case();  // has negatives; force the radix path
    const DiffResult d = run_mutant(c, mutant::SortMutation::kRadixSignBug, true);
    CHECK_FALSE(d.equal);
    MESSAGE("kRadixSignBug caught: " << d.message);
}

TEST_CASE("MUTATION: kUnstableTiebreak (reversed tie order) is CAUGHT vs ref") {
    const Case c = tie_case();
    // DuckDB's tie order is undefined, so the stability check is vs the reference
    // only (also_duckdb=false): real==reference (both stable), mutant reverses.
    const DiffResult d =
        run_mutant(c, mutant::SortMutation::kUnstableTiebreak, false);
    CHECK_FALSE(d.equal);
    MESSAGE("kUnstableTiebreak caught: " << d.message);
}

TEST_CASE("MUTATION: kEmitTailOffByOne (dropped last row) is CAUGHT") {
    const Case c = desc_case();
    const DiffResult d =
        run_mutant(c, mutant::SortMutation::kEmitTailOffByOne, true);
    CHECK_FALSE(d.equal);
    MESSAGE("kEmitTailOffByOne caught: " << d.message);
}
