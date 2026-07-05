//  WP-6 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken JOIN operators
//  (ops/join_mutants.{h,cpp}) — each a faithful copy of the real build / pair-gen
//  / output loop reusing the SAME ops/join_internal.h helpers, with exactly one
//  bad step — and SHOW the differential (independent reference + DuckDB when
//  staged) CATCHES every mutant, while the REAL operator PASSES the identical diff.
//
//  Planted mutants:
//    * kDropProbeMatch        — the §5 named bug: skip emitting one matched row.
//    * kLeftWrongNull         — a LEFT unmatched probe row emits build row 0's
//      real values instead of NULLs.
//    * kCompositeFirstKeyOnly — composite join matches on the first key alone.
//    * kFanoutTailOffByOne    — the final output batch is one row short.
//
//  Replay:  ./join_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/join.h"
#include "ops/join_mutants.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

struct Case {
    Table probe;
    Table build;
    JoinQuery query;
};

DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_join_reference(c.probe, c.build, c.query);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            const ResultSet dk = run_join_duckdb(c.probe, c.build, c.query);
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

DiffResult run_mutant(const Case& c, mutant::Mutation m, std::size_t bs) {
    auto ps = std::make_unique<Scan>(c.probe, bs);
    auto bsn = std::make_unique<Scan>(c.build, bs);
    mutant::HashJoin j(std::move(ps), std::move(bsn), c.query.probe_keys,
                       c.query.build_keys, c.query.type, m);
    return diff_tree_vs_oracles(j, c);
}

DiffResult run_real(const Case& c, std::size_t bs) {
    auto tree = build_join_pipeline(c.probe, c.build, c.query, bs);
    return diff_tree_vs_oracles(*tree, c);
}

// Single-key INNER case with real matches (1 -> two build rows, 2 -> one).
Case single_key_inner() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2}));
    bcols.push_back(i64_col({10, 11, 20}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 3}));  // 1,2 match; 3 misses
    JoinQuery q{{0}, {0}, JoinType::Inner};
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            std::move(q)};
}

// Single-key LEFT case with an unmatched probe row (key 3).
Case single_key_left() {
    Case c = single_key_inner();
    c.query.type = JoinType::Left;
    return c;
}

// Composite-key INNER case where first-key-only over-matches.
Case composite_inner() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I32);
    bs.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1}));        // first key same
    bcols.push_back(i32_col({10, 20}));      // second key differs
    bcols.push_back(i64_col({100, 200}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1}));   // (1,20) matches exactly ONE build row...
    pcols.push_back(i32_col({20}));  // ...but first-key-only would match BOTH
    JoinQuery q{{0, 1}, {0, 1}, JoinType::Inner};
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            std::move(q)};
}

// High-fanout INNER case: one probe row of key 5 fans out to 3 build rows, so the
// (single) output batch has m>1 and the tail-off-by-one mutant drops a row.
Case fanout_inner() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({5, 5, 5}));
    bcols.push_back(i64_col({1, 2, 3}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({5}));
    JoinQuery q{{0}, {0}, JoinType::Inner};
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            std::move(q)};
}

}  // namespace

TEST_CASE("MUTATION: the REAL join passes the differential") {
    for (std::size_t bs : {64u, 2048u}) {
        CHECK(run_real(single_key_inner(), bs).equal);
        CHECK(run_real(single_key_left(), bs).equal);
        CHECK(run_real(composite_inner(), bs).equal);
        CHECK(run_real(fanout_inner(), bs).equal);
    }
}

TEST_CASE("MUTATION: kDropProbeMatch (skip a matched row) is CAUGHT") {
    const DiffResult d =
        run_mutant(single_key_inner(), mutant::Mutation::kDropProbeMatch, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kDropProbeMatch caught: " << d.message);
}

TEST_CASE("MUTATION: kLeftWrongNull (unmatched emits real build row) is CAUGHT") {
    const DiffResult d =
        run_mutant(single_key_left(), mutant::Mutation::kLeftWrongNull, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kLeftWrongNull caught: " << d.message);
}

TEST_CASE("MUTATION: kCompositeFirstKeyOnly (over-match) is CAUGHT") {
    const DiffResult d = run_mutant(
        composite_inner(), mutant::Mutation::kCompositeFirstKeyOnly, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kCompositeFirstKeyOnly caught: " << d.message);
}

TEST_CASE("MUTATION: kFanoutTailOffByOne (short final batch) is CAUGHT") {
    const DiffResult d =
        run_mutant(fanout_inner(), mutant::Mutation::kFanoutTailOffByOne, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kFanoutTailOffByOne caught: " << d.message);
}
