//  WP-12 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken AS-OF operators
//  (tsx/asof_mutants.{h,cpp}) — each a faithful copy of the real build / merge /
//  emit loop reusing the SAME ops/join_internal.h gather emitters + frozen
//  Sort/HashTable, with exactly one bad step — and SHOW the differential
//  (independent brute-force reference + DuckDB ASOF JOIN when staged) CATCHES every
//  mutant, while the REAL operator PASSES the identical diff.
//
//  Planted mutants:
//    * kBoundaryStrict   — `<` instead of `<=`: drops equal-timestamp matches (the
//      classic `>`-vs-`>=` as-of boundary bug).
//    * kNearestFollowing — emits the nearest FOLLOWING build row, not the preceding.
//    * kIgnoreLastKey    — composite as-of matches on all-but-the-last key (over-join).
//    * kLeftWrongNull    — a LEFT unmatched probe row emits build row 0's real values.
//
//  Replay:  ./asof_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"
#include "tsx/asof.h"
#include "tsx/asof_mutants.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

OwnedColumn ts_col(const std::vector<std::int64_t>& v,
                   const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::TS, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}
OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

struct Case {
    Table probe;
    Table build;
    std::vector<std::uint32_t> lk, rk;
    std::uint32_t lt, rt;
    tsx::AsofType type;
    std::optional<std::int64_t> tol;
};

Plan case_plan(const Case& c) {
    std::vector<ColRef> l, r;
    for (auto i : c.lk) l.push_back(ColRef(static_cast<int>(i)));
    for (auto i : c.rk) r.push_back(ColRef(static_cast<int>(i)));
    return scan(c.probe)
        .asof_join(scan(c.build), l, r, ColRef(static_cast<int>(c.lt)),
                   ColRef(static_cast<int>(c.rt)), c.type, c.tol)
        .plan();
}

DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_asof_reference(c.probe, c.build, c.lk, c.rk, c.lt,
                                             c.rt, c.type, c.tol);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            const ResultSet dk = run_plan_duckdb(case_plan(c));
            d = compare_result_sets(engine, dk);
        } catch (const DuckDBError&) { /* divergence backstop; ignore */ }
    }
    return d;
}

DiffResult run_mutant(const Case& c, mutant::AsofMutation m, std::size_t bs) {
    auto ps = std::make_unique<Scan>(c.probe, bs);
    auto bsn = std::make_unique<Scan>(c.build, bs);
    mutant::AsofJoin j(std::move(ps), std::move(bsn), c.lk, c.rk, c.lt, c.rt,
                       c.type, c.tol, m);
    return diff_tree_vs_oracles(j, c);
}

DiffResult run_real(const Case& c, std::size_t bs) {
    auto ps = std::make_unique<Scan>(c.probe, bs);
    auto bsn = std::make_unique<Scan>(c.build, bs);
    tsx::AsofJoin j(std::move(ps), std::move(bsn), c.lk, c.rk, c.lt, c.rt, c.type,
                    c.tol);
    return diff_tree_vs_oracles(j, c);
}

// Single-key INNER with a boundary tie (probe t=20 == build t=20), an interior
// nearest-preceding (probe t=25 -> build t=20), and a following row (t=30) so the
// preceding/following mutants differ.
Case boundary_case() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::TS);
    bs.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 1}));
    bcols.push_back(ts_col({10, 20, 30}));
    bcols.push_back(i64_col({100, 200, 300}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1}));
    pcols.push_back(ts_col({20, 25}));  // 20: exact tie; 25: nearest preceding=20
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            {0}, {0}, 1, 1, tsx::AsofType::Inner, std::nullopt};
}

// Single-key LEFT with an unmatched probe row (t=5 before all build rows).
Case left_case() {
    Case c = boundary_case();
    c.type = tsx::AsofType::Left;
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1}));
    pcols.push_back(ts_col({5, 25}));  // 5 < min build => unmatched; 25 -> 20
    c.probe = Table(c.probe.schema(), std::move(pcols));
    return c;
}

// Composite-key INNER where first-key-only would over-match: build (1,7)@t and
// (1,9)@t are distinct partitions; probe (1,9) must match only the (1,9) build row.
Case composite_case() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I32);
    bs.fields.emplace_back("c2", Type::TS);
    bs.fields.emplace_back("c3", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1}));
    bcols.push_back(i32_col({7, 9}));
    bcols.push_back(ts_col({10, 50}));  // (1,7)@10 ; (1,9)@50
    bcols.push_back(i64_col({700, 900}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I32);
    ps.fields.emplace_back("c2", Type::TS);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1}));
    pcols.push_back(i32_col({9}));
    pcols.push_back(ts_col({20}));  // (1,9): only build (1,9)@50 has t>20 -> NO match;
                                    // first-key-only would wrongly match (1,7)@10.
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            {0, 1}, {0, 1}, 2, 2, tsx::AsofType::Inner, std::nullopt};
}

}  // namespace

TEST_CASE("MUTATION: the REAL as-of join passes the differential") {
    for (std::size_t bs : {64u, 2048u}) {
        CHECK(run_real(boundary_case(), bs).equal);
        CHECK(run_real(left_case(), bs).equal);
        CHECK(run_real(composite_case(), bs).equal);
    }
}

TEST_CASE("MUTATION: kBoundaryStrict (drops equal-timestamp match) is CAUGHT") {
    const DiffResult d =
        run_mutant(boundary_case(), mutant::AsofMutation::kBoundaryStrict, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kBoundaryStrict caught: " << d.message);
}

TEST_CASE("MUTATION: kNearestFollowing (forward instead of backward) is CAUGHT") {
    const DiffResult d =
        run_mutant(boundary_case(), mutant::AsofMutation::kNearestFollowing, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kNearestFollowing caught: " << d.message);
}

TEST_CASE("MUTATION: kIgnoreLastKey (composite over-match) is CAUGHT") {
    const DiffResult d =
        run_mutant(composite_case(), mutant::AsofMutation::kIgnoreLastKey, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kIgnoreLastKey caught: " << d.message);
}

TEST_CASE("MUTATION: kLeftWrongNull (unmatched emits real build row) is CAUGHT") {
    const DiffResult d =
        run_mutant(left_case(), mutant::AsofMutation::kLeftWrongNull, 2048);
    CHECK_FALSE(d.equal);
    MESSAGE("kLeftWrongNull caught: " << d.message);
}
