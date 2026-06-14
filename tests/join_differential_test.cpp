//  WP-6 / M3 PROOF: the hash equi-join diffs GREEN against the oracle on seeded
//  random two-table join queries, plus targeted edge cases. The always-available
//  independent reference oracle (nested-loop / multimap, run_join_reference) runs
//  every case; when the DuckDB amalgamation is staged (QE_WITH_DUCKDB) the SAME
//  runner + comparator also diff against DuckDB, the AUTHORITATIVE golden model
//  (D16). Join output has no inherent row order, so the comparator canonicalizes
//  (sorts both sides on all output columns) before diffing (D12); F64 payload
//  columns compare within the D11 epsilon (they are carried verbatim, so exact).
//
//  Determinism: every run prints its seed (tests/wp1_test_main.cpp). Replay:
//      ./join_differential_test --seed N
//  Batch size is varied across {64,256,2048} so build-across-batch-boundaries and
//  the output-batch fan-out tail (a single probe row fanning out past 2048 rows)
//  are exercised.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/join.h"
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
using namespace qe::ops_test;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// Diff against BOTH the independent reference and (when staged) DuckDB.
DiffResult join_diff_all(const Table& p, const Table& b, const JoinQuery& q,
                         std::size_t bs) {
    DiffResult ref = run_join_vs_reference(p, b, q, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_join_differential(p, b, q, run_join_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            MESSAGE("DuckDB raised (skipped, not a diff): " << e.what());
        }
    }
    return ref;
}

// --- small typed column builders for the edge cases -------------------------
OwnedColumn i64_col(const std::vector<std::int64_t>& v,
                    const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

}  // namespace

TEST_CASE("M3: join diffs green vs oracle on random two-table queries") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x301Du);

    for (int iter = 0; iter < 200; ++iter) {
        JoinCase jc = gen_join_case(rng);
        const std::size_t bs = kBatchSizes[rng() % 3];
        const DiffResult d =
            join_diff_all(jc.probe, jc.build, jc.query, bs);
        CHECK(d.equal);
        if (!d.equal) {
            MESSAGE("DIFF at iter=" << iter
                                    << " probe_rows=" << jc.probe.num_rows()
                                    << " build_rows=" << jc.build.num_rows()
                                    << " keys=" << jc.query.probe_keys.size()
                                    << " type="
                                    << (jc.query.type == JoinType::Left ? "LEFT"
                                                                        : "INNER")
                                    << " batch=" << bs << " : " << d.message);
            break;  // first failure suffices; replay with --seed
        }
    }
}

// Build/probe with one i32 key column. Probe keys: 1 matches (k build rows), 2
// matches, 3 has NO build match, NULL key never matches.
namespace {

struct OneKey {
    Table probe;
    Table build;
};

OneKey one_key_case() {
    // build: key c0 (i32, one NULL), payload c1 (i64).
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2, 5, 5, 5}, /*nulls=*/{}));    // key
    bcols.push_back(i64_col({10, 11, 20, 50, 51, 52}));           // payload
    Table build(bs, std::move(bcols));

    // probe: key c0 (i32, one NULL at row 4), payload c1 (i64).
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 3, 5, 0}, /*nulls=*/{4}));  // 1,2 match; 3 miss
    pcols.push_back(i64_col({100, 200, 300, 500, 999}));
    Table probe(ps, std::move(pcols));
    return {std::move(probe), std::move(build)};
}

}  // namespace

TEST_CASE("edge: single key — inner & left, matches/no-match/null key") {
    OneKey c = one_key_case();
    SUBCASE("inner") {
        JoinQuery q{{0}, {0}, JoinType::Inner};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(c.probe, c.build, q, bs).equal);
    }
    SUBCASE("left") {
        JoinQuery q{{0}, {0}, JoinType::Left};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(c.probe, c.build, q, bs).equal);
    }
}

TEST_CASE("edge: composite key (i32, f64) — inner & left") {
    // build: c0 i32 key, c1 f64 key, c2 i64 payload.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::F64);
    bs.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2, 2}));
    bcols.push_back(f64_col({1.0, 2.0, 1.0, 2.0}));
    bcols.push_back(i64_col({11, 12, 21, 22}));
    Table build(bs, std::move(bcols));

    // probe: (1,2.0) matches one build row; (1,9.0) no match; (2,1.0) matches.
    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1, 2, 2}));
    pcols.push_back(f64_col({2.0, 9.0, 1.0, 2.0}));
    Table probe(ps, std::move(pcols));

    SUBCASE("inner") {
        JoinQuery q{{0, 1}, {0, 1}, JoinType::Inner};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
    SUBCASE("left") {
        JoinQuery q{{0, 1}, {0, 1}, JoinType::Left};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}

TEST_CASE("edge: empty build side — inner (0 rows) & left (all NULL build)") {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({}));
    bcols.push_back(i64_col({}));
    Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 3}));
    Table probe(ps, std::move(pcols));

    SUBCASE("inner") {
        JoinQuery q{{0}, {0}, JoinType::Inner};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
    SUBCASE("left") {
        JoinQuery q{{0}, {0}, JoinType::Left};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}

TEST_CASE("edge: empty probe side — inner & left both empty") {
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 2, 3}));
    Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({}));
    pcols.push_back(i64_col({}));
    Table probe(ps, std::move(pcols));

    for (JoinType t : {JoinType::Inner, JoinType::Left}) {
        JoinQuery q{{0}, {0}, t};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}

TEST_CASE("edge: all-match — every probe row matches every build row (1 key)") {
    // build & probe all share key 7 -> a full cross of payloads.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({7, 7, 7, 7}));
    bcols.push_back(i64_col({1, 2, 3, 4}));
    Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({7, 7, 7}));
    Table probe(ps, std::move(pcols));

    for (JoinType t : {JoinType::Inner, JoinType::Left}) {
        JoinQuery q{{0}, {0}, t};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}

TEST_CASE("edge: high-fanout key crosses the 2048 output-batch tail") {
    // One build key (7) repeated 3000 times: a single probe row of key 7 fans out
    // to 3000 output rows, forcing >=2 output batches and a non-trivial tail.
    const int N = 3000;
    std::vector<std::int32_t> bk(N, 7);
    std::vector<std::int64_t> bv(N);
    for (int i = 0; i < N; ++i) bv[i] = i;
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col(bk));
    bcols.push_back(i64_col(bv));
    Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({7, 9, 7}));        // two fan-out rows + one miss
    pcols.push_back(i64_col({100, 200, 300}));
    Table probe(ps, std::move(pcols));

    for (JoinType t : {JoinType::Inner, JoinType::Left}) {
        JoinQuery q{{0}, {0}, t};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}

TEST_CASE("edge: NULL keys on both sides never match (incl. NULL==NULL)") {
    // Both sides have NULL keys; a NULL must NOT join a NULL (kNeverMatch).
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 0, 2}, /*nulls=*/{1}));  // row1 NULL key
    bcols.push_back(i64_col({10, 11, 12}));
    Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 0, 0}, /*nulls=*/{1, 2}));  // two NULL keys
    Table probe(ps, std::move(pcols));

    for (JoinType t : {JoinType::Inner, JoinType::Left}) {
        JoinQuery q{{0}, {0}, t};
        for (std::size_t bs : kBatchSizes)
            CHECK(join_diff_all(probe, build, q, bs).equal);
    }
}
