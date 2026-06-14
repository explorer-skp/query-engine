//  WP-3 unit tests: the mechanics of Scan / Filter / Project in isolation —
//  batch boundaries & tails, zero-copy validity sub-views, null handling,
//  all-pass / all-fail / empty results, and (critically) Filter/Project over an
//  input batch that ALREADY carries a selection vector. The big seeded
//  engine-vs-oracle differential lives in oracle_differential_test.cpp.
//
//  Replay:  ./ops_test --seed N

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/filter.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::ops_test;
using qe::oracle::drain_operator;
using qe::oracle::ResultSet;

namespace {

Schema schema_i32_f64() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    return s;
}

// Table: c0 = 0..n-1, c1 = c0 * 1.5, every 7th c0 is NULL.
Table seq_table(std::size_t n) {
    std::vector<std::int32_t> a(n);
    std::vector<double> b(n);
    std::vector<std::size_t> nulls;
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::int32_t>(i);
        b[i] = static_cast<double>(i) * 1.5;
        if (i % 7 == 0) nulls.push_back(i);
    }
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(a, nulls));
    cols.push_back(f64_col(b));
    return Table(schema_i32_f64(), std::move(cols));
}

}  // namespace

TEST_CASE("scan yields every row across batch boundaries incl. the tail") {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{64}, std::size_t{65}, std::size_t{200},
                          std::size_t{5000}}) {
        Table t = seq_table(n);
        Scan scan(t, 64);
        ResultSet rs = drain_operator(scan);
        CHECK(rs.num_rows() == n);
        CHECK(rs.num_cols() == 2);
        // Order is preserved by scan; check a few rows' values & nullness.
        for (std::size_t i = 0; i < n; ++i) {
            const auto& c0 = rs.rows[i][0];
            const auto& c1 = rs.rows[i][1];
            if (i % 7 == 0) {
                CHECK(c0.is_null);
            } else {
                CHECK_FALSE(c0.is_null);
                CHECK(c0.i == static_cast<std::int64_t>(i));
            }
            CHECK_FALSE(c1.is_null);
            CHECK(c1.f == doctest::Approx(static_cast<double>(i) * 1.5));
        }
    }
}

TEST_CASE("scan reports correct batch sizes and tail") {
    Table t = seq_table(5000);
    Scan scan(t, 2048);
    scan.open();
    std::vector<std::size_t> sizes;
    while (auto b = scan.next()) sizes.push_back(b->row_count);
    scan.close();
    CHECK(sizes == std::vector<std::size_t>{2048, 2048, 904});
}

TEST_CASE("scan rejects a non-64-multiple batch size") {
    Table t = seq_table(10);
    CHECK_THROWS([&] { Scan s(t, 100); }());  // 100 is not a multiple of 64
}

TEST_CASE("filter: all-pass, all-fail, empty, and NULL predicate excludes") {
    Table t = seq_table(300);

    SUBCASE("predicate TRUE keeps all (minus nothing)") {
        auto scan = std::make_unique<Scan>(t, 64);
        Filter f(std::move(scan), lit(Scalar::boolean(true)));
        ResultSet rs = drain_operator(f);
        CHECK(rs.num_rows() == 300);
    }
    SUBCASE("predicate FALSE keeps none (exhausted -> nullopt)") {
        auto scan = std::make_unique<Scan>(t, 64);
        Filter f(std::move(scan), lit(Scalar::boolean(false)));
        ResultSet rs = drain_operator(f);
        CHECK(rs.num_rows() == 0);
    }
    SUBCASE("c0 >= 100 keeps the upper rows; NULL c0 rows excluded") {
        auto scan = std::make_unique<Scan>(t, 64);
        Filter f(std::move(scan), ge(col(Type::I32, 0), lit(Scalar::i32(100))));
        ResultSet rs = drain_operator(f);
        // Rows 100..299 minus the NULL c0 rows (i%7==0) in that range.
        std::size_t expected = 0;
        for (std::size_t i = 100; i < 300; ++i)
            if (i % 7 != 0) ++expected;
        CHECK(rs.num_rows() == expected);
    }
}

TEST_CASE("filter & project honor an input batch's selection vector") {
    // One owned batch of 5 i32 rows, selected/reordered to logical [50,30,10].
    OwnedBatch ob;
    ob.add_column(i32_col({10, 20, 30, 40, 50}));
    ob.set_selection({4, 2, 0});  // -> 50, 30, 10
    Schema s;
    s.fields.emplace_back("c0", Type::I32);

    SUBCASE("filter c0 > 15 over the selected batch keeps {50,30}") {
        auto src = std::make_unique<OneBatchSource>(s, ob);
        Filter f(std::move(src), gt(col(Type::I32, 0), lit(Scalar::i32(15))));
        ResultSet rs = drain_operator(f);
        CHECK(rs.num_rows() == 2);
        std::vector<std::int64_t> got{rs.rows[0][0].i, rs.rows[1][0].i};
        std::sort(got.begin(), got.end());
        CHECK(got == std::vector<std::int64_t>{30, 50});
    }
    SUBCASE("project (c0, c0+1) over the selected batch") {
        auto src = std::make_unique<OneBatchSource>(s, ob);
        std::vector<Projection> projs;
        projs.push_back({"p0", col(Type::I32, 0)});
        projs.push_back({"p1", add(col(Type::I32, 0), lit(Scalar::i32(1)))});
        Project p(std::move(src), std::move(projs));
        ResultSet rs = drain_operator(p);
        CHECK(rs.num_rows() == 3);
        // Logical order [50,30,10] is preserved (no canonicalization here).
        CHECK(rs.rows[0][0].i == 50);
        CHECK(rs.rows[0][1].i == 51);
        CHECK(rs.rows[1][0].i == 30);
        CHECK(rs.rows[1][1].i == 31);
        CHECK(rs.rows[2][0].i == 10);
        CHECK(rs.rows[2][1].i == 11);
    }
}

TEST_CASE("project propagates nulls and computes derived columns") {
    Table t = seq_table(100);
    auto scan = std::make_unique<Scan>(t, 64);
    std::vector<Projection> projs;
    projs.push_back({"p0", col(Type::I32, 0)});
    projs.push_back({"p1", mul(col(Type::I32, 0), lit(Scalar::i32(2)))});
    Project p(std::move(scan), std::move(projs));
    ResultSet rs = drain_operator(p);
    CHECK(rs.num_rows() == 100);
    for (std::size_t i = 0; i < 100; ++i) {
        if (i % 7 == 0) {
            CHECK(rs.rows[i][0].is_null);
            CHECK(rs.rows[i][1].is_null);  // null propagates through *2
        } else {
            CHECK(rs.rows[i][0].i == static_cast<std::int64_t>(i));
            CHECK(rs.rows[i][1].i == static_cast<std::int64_t>(i) * 2);
        }
    }
}
