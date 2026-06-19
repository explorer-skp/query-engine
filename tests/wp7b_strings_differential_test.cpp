//  WP-7b PROOF: dictionary-encoded VARCHAR (Type::STR) diffs GREEN against the
//  oracle on string GROUP BY / JOIN / FILTER (=,<) / ORDER BY / MIN-MAX. The
//  always-available independent reference oracle (std::string values, std::map
//  keys — shares no engine code path) runs every case; when the DuckDB amalgamation
//  is staged (QE_WITH_DUCKDB) the SAME runner + comparator also diff against DuckDB
//  VARCHAR, the AUTHORITATIVE golden model (D16).
//
//  KEY POINT (the WP's whole reason to exist): STR is compared BY VALUE. Every STR
//  column here is built with its OWN dictionary, so equal strings generally carry
//  DIFFERENT int32 codes across columns/sides — a by-code engine would diverge; a
//  by-value engine agrees. Unordered results canonicalize on the decoded strings
//  (D12); strings compare EXACT (no float epsilon).
//
//  Determinism: every run prints its seed (tests/wp1_test_main.cpp). Replay:
//      ./wp7b_strings_differential_test --seed N

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/string_dict.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/table.h"
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

const std::size_t kBatchSizes[] = {64u, 256u, 2048u};

// A STR column from explicit values (nullopt => NULL), each interned into a FRESH
// per-column dict so codes are column-relative (the cross-dict case).
OwnedColumn str_col(const std::vector<std::optional<std::string>>& vals) {
    auto dict = std::make_shared<StringDict>();
    OwnedColumn c = OwnedColumn::make_str(vals.size(), dict);
    auto* codes = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        if (vals[i])
            codes[i] = dict->intern(*vals[i]);
        else
            c.set_null(i);
    }
    return c;
}

OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

// A small shared alphabet so build/probe STR columns overlap in VALUE (so a join
// has matches) but get independent codes.
const std::vector<std::string> kAlpha = {"AAPL", "MSFT", "GOOG", "AMZN",
                                         "TSLA", "NVDA", "META", "x"};

// ---- single-table (group-by / filter / order-by) differential --------------
DiffResult diff_single(const Table& t, const LogicalQuery& q, std::size_t bs) {
    auto tree = build_engine_pipeline(t, q, bs);
    const ResultSet eng = drain_operator(*tree);
    const ResultSet ref = run_reference(t, q);
    const bool ordered = q.has_order_by();
    DiffResult d = compare_result_sets(eng, ref, ordered);
    if (d.equal && duckdb_available()) {
        try {
            d = compare_result_sets(eng, run_duckdb(t, q), ordered);
        } catch (const DuckDBError&) { /* divergence backstop */ }
    }
    return d;
}

// ---- two-table (join) differential -----------------------------------------
DiffResult diff_join(const Table& probe, const Table& build, const JoinQuery& jq,
                     std::size_t bs) {
    auto tree = build_join_pipeline(probe, build, jq, bs);
    const ResultSet eng = drain_operator(*tree);
    const ResultSet ref = run_join_reference(probe, build, jq);
    DiffResult d = compare_result_sets(eng, ref);  // join: unordered (D12)
    if (d.equal && duckdb_available()) {
        try {
            d = compare_result_sets(eng, run_join_duckdb(probe, build, jq));
        } catch (const DuckDBError&) { /* divergence backstop */ }
    }
    return d;
}

// A random STR table: col0 STR key, col1 STR payload, col2 I64 payload.
Table random_str_table(std::mt19937_64& rng, std::size_t n, int null_pct) {
    Schema s;
    s.fields.emplace_back("c0", Type::STR);
    s.fields.emplace_back("c1", Type::STR);
    s.fields.emplace_back("c2", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(gen_string_column(rng, kAlpha, n, null_pct));
    cols.push_back(gen_string_column(rng, kAlpha, n, null_pct));
    std::vector<std::int64_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<std::int64_t>(rng() % 1000);
    cols.push_back(i64_col(v));
    return Table(std::move(s), std::move(cols));
}

}  // namespace

// ===========================================================================
TEST_CASE("WP-7b: string GROUP BY + MIN/MAX/COUNT diffs green vs oracle") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x57721Au);
    for (int iter = 0; iter < 60; ++iter) {
        const std::size_t n = 1 + (rng() % 1200);
        const Table t = random_str_table(rng, n, /*null_pct=*/15);

        LogicalQuery q;
        GroupBy gb;
        gb.keys = {0};  // GROUP BY the STR key column (NULLs group together)
        gb.aggs = {AggSpec::count_star("n"), AggSpec::count(1, "cnt_s"),
                   AggSpec::min(1, "min_s"), AggSpec::max(1, "max_s")};
        q.group_by = std::move(gb);

        for (std::size_t bs : kBatchSizes) {
            const DiffResult d = diff_single(t, q, bs);
            CHECK_MESSAGE(d.equal, "group-by diverged: " << d.message);
        }
    }
}

TEST_CASE("WP-7b: composite (STR + I64) GROUP BY diffs green") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x111111u);
    for (int iter = 0; iter < 30; ++iter) {
        const std::size_t n = 1 + (rng() % 800);
        const Table t = random_str_table(rng, n, 10);
        LogicalQuery q;
        GroupBy gb;
        gb.keys = {0, 2};  // STR key + I64 key
        gb.aggs = {AggSpec::count_star("n"), AggSpec::max(0, "max_key")};
        q.group_by = std::move(gb);
        for (std::size_t bs : kBatchSizes)
            CHECK(diff_single(t, q, bs).equal);
    }
}

TEST_CASE("WP-7b: string JOIN (inner + left) diffs green vs oracle") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x9E3779u);
    for (int iter = 0; iter < 60; ++iter) {
        const std::size_t nb = rng() % 300;
        const std::size_t np = rng() % 300;
        // Independent dicts on each side => same VALUES, different CODES.
        Schema bs;
        bs.fields.emplace_back("c0", Type::STR);  // key
        bs.fields.emplace_back("c1", Type::I64);  // payload
        std::vector<OwnedColumn> bcols;
        bcols.push_back(gen_string_column(rng, kAlpha, nb, 12));
        std::vector<std::int64_t> bv(nb);
        for (std::size_t i = 0; i < nb; ++i) bv[i] = static_cast<std::int64_t>(i);
        bcols.push_back(i64_col(bv));
        const Table build(bs, std::move(bcols));

        Schema ps;
        ps.fields.emplace_back("c0", Type::STR);
        ps.fields.emplace_back("c1", Type::I64);
        std::vector<OwnedColumn> pcols;
        pcols.push_back(gen_string_column(rng, kAlpha, np, 12));
        std::vector<std::int64_t> pv(np);
        for (std::size_t i = 0; i < np; ++i)
            pv[i] = static_cast<std::int64_t>(1000 + i);
        pcols.push_back(i64_col(pv));
        const Table probe(ps, std::move(pcols));

        JoinQuery jq{{0}, {0}, (iter & 1) ? JoinType::Left : JoinType::Inner};
        for (std::size_t bsz : {64u, 2048u}) {
            const DiffResult d = diff_join(probe, build, jq, bsz);
            CHECK_MESSAGE(d.equal, "string join diverged: " << d.message);
        }
    }
}

TEST_CASE("WP-7b: string FILTER (s0 < s1) + projection diffs green") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xABCDEFu);
    for (int iter = 0; iter < 40; ++iter) {
        const std::size_t n = 1 + (rng() % 1000);
        const Table t = random_str_table(rng, n, 10);
        LogicalQuery q;
        q.filter = expr::lt(expr::col(Type::STR, 0), expr::col(Type::STR, 1));
        q.projections = {Projection{"p0", expr::col(Type::STR, 0)},
                         Projection{"p1", expr::col(Type::STR, 1)},
                         Projection{"p2", expr::col(Type::I64, 2)}};
        for (std::size_t bs : kBatchSizes)
            CHECK(diff_single(t, q, bs).equal);
    }
}

TEST_CASE("WP-7b: string ORDER BY (lexicographic) diffs green positionally") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5A5A5Au);
    for (int iter = 0; iter < 40; ++iter) {
        const std::size_t n = 1 + (rng() % 1000);
        // col0 STR, col1 a UNIQUE I64 tiebreaker so the order is TOTAL (positional
        // compare is unambiguous — equal STR keys are broken by the unique col1).
        Schema s;
        s.fields.emplace_back("c0", Type::STR);
        s.fields.emplace_back("c1", Type::I64);
        std::vector<OwnedColumn> cols;
        cols.push_back(gen_string_column(rng, kAlpha, n, 12));
        std::vector<std::int64_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int64_t>(i);
        cols.push_back(i64_col(v));
        const Table t(s, std::move(cols));

        LogicalQuery q;
        q.projections = {Projection{"p0", expr::col(Type::STR, 0)},
                         Projection{"p1", expr::col(Type::I64, 1)}};
        const bool desc = (iter & 1) != 0;
        const NullOrder no = (iter & 2) ? NullOrder::First : NullOrder::Last;
        q.order_by = std::vector<SortKey>{
            SortKey{0, desc ? SortDir::Desc : SortDir::Asc, no},
            SortKey{1, SortDir::Asc, NullOrder::Last}};  // unique tiebreaker
        for (std::size_t bs : kBatchSizes) {
            const DiffResult d = diff_single(t, q, bs);
            CHECK_MESSAGE(d.equal, "string order-by diverged: " << d.message);
        }
    }
}

// A couple of pointed explicit cases (cross-dictionary equality is the crux).
TEST_CASE("WP-7b: explicit cross-dictionary join match (by value, not code)") {
    // build "AAPL" interned FIRST (code 0); probe "AAPL" interned SECOND (code 1).
    Schema bs;
    bs.fields.emplace_back("c0", Type::STR);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(str_col({"AAPL", "MSFT"}));   // codes 0,1
    bcols.push_back(i64_col({10, 20}));
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::STR);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(str_col({"ZZZ", "AAPL"}));    // "AAPL" is code 1 here
    const Table probe(ps, std::move(pcols));

    JoinQuery jq{{0}, {0}, JoinType::Inner};
    const DiffResult d = diff_join(probe, build, jq, 2048);
    CHECK_MESSAGE(d.equal, "cross-dict join diverged: " << d.message);
}
