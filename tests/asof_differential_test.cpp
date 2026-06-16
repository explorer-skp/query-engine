//  WP-12: the backward AS-OF join differential. ONE Plan (scan(probe).asof_join(
//  scan(build), ...)) drives BOTH backends: it lowers to the tsx::AsofJoin operator
//  tree and renders to the DuckDB `ASOF [LEFT] JOIN` SQL. Each generated tick-like
//  case diffs GREEN vs the independent brute-force reference (always) AND, when
//  staged (QE_WITH_DUCKDB), the AUTHORITATIVE DuckDB diff. As-of output is UNORDERED
//  (no ORDER BY wraps it), so the comparator canonicalizes (D12); float tolerance is
//  per D11 (the generators emit finite, moderate floats only on key/payload columns).
//
//  Coverage: irregular timestamps with gaps (probe before any build row => no-match),
//  EXACT-timestamp ties at the `>=` boundary, single / composite / ZERO partition
//  keys, INNER & LEFT, NULL keys & NULL timestamps (never match), the within-
//  tolerance variant (both INNER and LEFT), and empty sides. Batch size is swept
//  across {64,256,2048} so the merge cursor straddles batch boundaries.
//
//  Determinism: build-side (key, timestamp) is UNIQUE (globally-distinct build
//  timestamps), so the nearest-preceding pick is unambiguous and the diff is never
//  flaky. Seed printed (tests/wp1_test_main.cpp). Replay:
//      ./asof_differential_test --seed N

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"
#include "tsx/asof.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// A TS column from int64 values; indices in `nulls` are marked NULL.
OwnedColumn ts_or_null(const std::vector<std::int64_t>& v,
                       const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::TS, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

// Build the as-of Plan from a generated case (the SINGLE source of truth feeding
// both backends). `probe`/`build` must outlive the returned Plan (Scan borrows).
Plan asof_plan(const Table& probe, const Table& build,
               const std::vector<std::uint32_t>& lk,
               const std::vector<std::uint32_t>& rk, std::uint32_t lt,
               std::uint32_t rt, tsx::AsofType type,
               std::optional<std::int64_t> tol) {
    std::vector<ColRef> l, r;
    for (auto i : lk) l.push_back(ColRef(static_cast<int>(i)));
    for (auto i : rk) r.push_back(ColRef(static_cast<int>(i)));
    return scan(probe)
        .asof_join(scan(build), l, r, ColRef(static_cast<int>(lt)),
                   ColRef(static_cast<int>(rt)), type, tol)
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

TEST_CASE("WP-12: random tick-like as-of joins diff green vs reference + DuckDB") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xA50Fu);
    for (int iter = 0; iter < 200; ++iter) {
        AsofCase c = gen_asof_case(rng);
        const Plan p =
            asof_plan(c.probe, c.build, c.left_keys, c.right_keys, c.left_time,
                      c.right_time, c.type, c.tolerance);
        CHECK(diff_all_batches(p));
    }
}

// ---- explicit edge shapes ----------------------------------------------------

namespace {

// build: key c0 (i32), timestamp c1 (TS), payload c2 (i64).
Table make_build() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::TS);
    s.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1, 2}));
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 4);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        const std::int64_t v[] = {10, 20, 30, 15};
        for (int i = 0; i < 4; ++i) d[i] = v[i];
        cols.push_back(std::move(t));
    }
    {
        OwnedColumn p = OwnedColumn::make(Type::I64, 4);
        auto* d = reinterpret_cast<std::int64_t*>(p.mutable_data());
        const std::int64_t v[] = {100, 200, 300, 150};
        for (int i = 0; i < 4; ++i) d[i] = v[i];
        cols.push_back(std::move(p));
    }
    return Table(s, std::move(cols));
}

// probe: key c0 (i32), timestamp c1 (TS). Includes a before-all gap (1,5), an exact
// boundary tie (1,10 and 2,15), an interior nearest-preceding (1,25), a no-key-match
// (3,99), and a late row (1,50). All keys/timestamps non-NULL (NULL handling is its
// own reference-only edge — DuckDB's ASOF NULL behavior diverges).
Table make_probe() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1, 2, 3, 1}));
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 6);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        const std::int64_t v[] = {5, 10, 25, 15, 99, 50};
        for (int i = 0; i < 6; ++i) d[i] = v[i];
        cols.push_back(std::move(t));
    }
    return Table(s, std::move(cols));
}

}  // namespace

TEST_CASE("WP-12 edge: single-key INNER (gap, boundary tie, nearest preceding)") {
    const Table build = make_build();
    const Table probe = make_probe();
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Inner, std::nullopt)));
}

TEST_CASE("WP-12 edge: single-key LEFT keeps unmatched probe rows (NULL build)") {
    const Table build = make_build();
    const Table probe = make_probe();
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Left, std::nullopt)));
}

TEST_CASE("WP-12 edge: within-tolerance INNER and LEFT (tol=10)") {
    const Table build = make_build();
    const Table probe = make_probe();
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Inner, std::int64_t{10})));
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Left, std::int64_t{10})));
}

TEST_CASE("WP-12 edge: zero partition keys (global backward as-of)") {
    // build: only a timestamp + payload; probe: only a timestamp.
    Schema bs;
    bs.fields.emplace_back("t", Type::TS);
    bs.fields.emplace_back("v", Type::I64);
    std::vector<OwnedColumn> bcols;
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 3);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        d[0] = 10; d[1] = 20; d[2] = 30;
        bcols.push_back(std::move(t));
    }
    {
        OwnedColumn v = OwnedColumn::make(Type::I64, 3);
        auto* d = reinterpret_cast<std::int64_t*>(v.mutable_data());
        d[0] = 100; d[1] = 200; d[2] = 300;
        bcols.push_back(std::move(v));
    }
    const Table build(bs, std::move(bcols));
    Schema ps;
    ps.fields.emplace_back("t", Type::TS);
    std::vector<OwnedColumn> pcols;
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 3);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        d[0] = 5; d[1] = 25; d[2] = 35;  // 5 before all (no match)
        pcols.push_back(std::move(t));
    }
    const Table probe(ps, std::move(pcols));
    CHECK(diff_all_batches(
        asof_plan(probe, build, {}, {}, 0, 0, tsx::AsofType::Inner, std::nullopt)));
    CHECK(diff_all_batches(
        asof_plan(probe, build, {}, {}, 0, 0, tsx::AsofType::Left, std::nullopt)));
}

TEST_CASE("WP-12 edge: empty build (INNER empty; LEFT all-NULL build cols)") {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({}));
    bcols.push_back(OwnedColumn::make(Type::TS, 0));
    const Table build(bs, std::move(bcols));
    const Table probe = make_probe();
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Inner, std::nullopt)));
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Left, std::nullopt)));
}

TEST_CASE("WP-12 edge: empty probe (no output)") {
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({}));
    pcols.push_back(OwnedColumn::make(Type::TS, 0));
    const Table probe(ps, std::move(pcols));
    const Table build = make_build();
    CHECK(diff_all_batches(asof_plan(probe, build, {0}, {0}, 1, 1,
                                     tsx::AsofType::Inner, std::nullopt)));
}

TEST_CASE("WP-12 edge: NULL keys & timestamps never match (engine == reference)") {
    // DOCUMENTED DIVERGENCES (excluded from the DuckDB-backed grammar; see the
    // generators.cpp note): DuckDB v1.1.3's ASOF matches a NULL probe timestamp to a
    // NULL build timestamp, and its NULL-KEY matching is data-dependent. This engine
    // takes the principled "a NULL key or NULL timestamp never matches" path. So we
    // check it ENGINE-vs-REFERENCE only (run_plan_vs_reference never calls DuckDB) —
    // both implement never-match.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::TS);
    bs.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2, 7}, /*nulls=*/{3}));        // (3): NULL key
    bcols.push_back(ts_or_null({10, 20, 0, 5}, /*nulls=*/{2}));   // (2): NULL ts
    {
        OwnedColumn p = OwnedColumn::make(Type::I64, 4);
        auto* d = reinterpret_cast<std::int64_t*>(p.mutable_data());
        d[0] = 100; d[1] = 200; d[2] = 999; d[3] = 777;
        bcols.push_back(std::move(p));
    }
    const Table build(bs, std::move(bcols));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::TS);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 2, 9}, /*nulls=*/{3}));         // (3): NULL key
    pcols.push_back(ts_or_null({25, 30, 0, 50}, /*nulls=*/{2}));   // (2): NULL ts
    const Table probe(ps, std::move(pcols));

    for (auto type : {tsx::AsofType::Inner, tsx::AsofType::Left}) {
        const Plan p =
            asof_plan(probe, build, {0}, {0}, 1, 1, type, std::nullopt);
        for (std::size_t bs2 : kBatchSizes)
            CHECK(run_plan_vs_reference(p, bs2).equal);  // reference-only
    }
}

TEST_CASE("WP-12 edge: composite key (two key columns)") {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I32);
    bs.fields.emplace_back("c2", Type::TS);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 1}));
    bcols.push_back(i32_col({7, 7, 9}));  // (1,7) and (1,9) distinct partitions
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 3);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        d[0] = 10; d[1] = 20; d[2] = 10;
        bcols.push_back(std::move(t));
    }
    const Table build(bs, std::move(bcols));
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I32);
    ps.fields.emplace_back("c2", Type::TS);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1}));
    pcols.push_back(i32_col({7, 9}));
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 2);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        d[0] = 25; d[1] = 25;  // (1,7)->t20 ; (1,9)->t10
        pcols.push_back(std::move(t));
    }
    const Table probe(ps, std::move(pcols));
    CHECK(diff_all_batches(asof_plan(probe, build, {0, 1}, {0, 1}, 2, 2,
                                     tsx::AsofType::Inner, std::nullopt)));
}
