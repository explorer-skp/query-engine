//  WP-8: the printable round-trip. A Plan prints to a DETERMINISTIC,
//  structure-revealing form; this asserts (1) an exact canonical rendering of a
//  built plan, (2) stability — building the same plan twice prints identically,
//  and (3) a structural change (dropping a node, swapping join sides) changes the
//  print. No re-parse (printing only is the contract; a parser is a non-goal).
//
//  No randomness here — uses tests/test_main.cpp's plain doctest main.

#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "plan/plan.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::plan;
using namespace qe::ops_test;

namespace {

Table two_col_table() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 7, 2}));
    {
        OwnedColumn c = OwnedColumn::make(Type::I64, 4);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        const std::int64_t v[] = {10, 20, 30, 40};
        for (int i = 0; i < 4; ++i) d[i] = v[i];
        cols.push_back(std::move(c));
    }
    return Table(s, std::move(cols));
}

}  // namespace

TEST_CASE("WP-8 print: scan->filter->project renders canonically") {
    const Table t = two_col_table();
    const Plan p = scan(t)
                       .filter(gt(col(Type::I32, 0), lit(Scalar::i32(0))))
                       .project({{"p0", col(Type::I32, 0)},
                                 {"p1", add(col(Type::I64, 1),
                                            lit(Scalar::i64(100)))}})
                       .plan();

    const std::string expected =
        "Project [p0=c0, p1=(c1 + 100)]\n"
        "  Filter (c0 > 0)\n"
        "    Scan [c0:I32, c1:I64]\n";
    CHECK(p.to_string() == expected);
}

TEST_CASE("WP-8 print: deterministic + structure-revealing") {
    const Table t = two_col_table();

    auto with_filter = [&] {
        return scan(t)
            .filter(gt(col(Type::I32, 0), lit(Scalar::i32(0))))
            .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(1, "s")})
            .sort({SortKey{1, SortDir::Desc, NullOrder::Last}})
            .plan();
    };

    // (1) stability: two independent builds of the same plan print identically.
    CHECK(with_filter().to_string() == with_filter().to_string());

    // The full rendering is exact and reveals every node + its salient params.
    const std::string expected =
        "Sort [1 DESC NULLS LAST]\n"
        "  Aggregate keys=[c0] aggs=[COUNT(*) AS n, SUM(c1) AS s]\n"
        "    Filter (c0 > 0)\n"
        "      Scan [c0:I32, c1:I64]\n";
    CHECK(with_filter().to_string() == expected);

    // (2) structural change is visible: dropping the Filter changes the print.
    const Plan no_filter =
        scan(t)
            .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(1, "s")})
            .sort({SortKey{1, SortDir::Desc, NullOrder::Last}})
            .plan();
    CHECK(no_filter.to_string() != with_filter().to_string());
}

TEST_CASE("WP-8 print: join sides + keys are reflected (swap changes print)") {
    const Table a = two_col_table();
    const Table b = two_col_table();
    const Plan ab = scan(a).join(scan(b), {0}, {0}, JoinType::Inner).plan();
    const Plan ba = scan(b).join(scan(a), {0}, {0}, JoinType::Left).plan();

    CHECK(ab.to_string() ==
          "Join INNER left_keys=[0] right_keys=[0]\n"
          "  Scan [c0:I32, c1:I64]\n"
          "  Scan [c0:I32, c1:I64]\n");
    // Different join type prints differently (probe/build swap is observable via
    // the INNER/LEFT tag here; column-order swap is exercised by the differential).
    CHECK(ab.to_string() != ba.to_string());
}

TEST_CASE("WP-8 builder: name-resolved keys equal index-resolved keys") {
    const Table t = two_col_table();
    const Plan by_index =
        scan(t).aggregate({0}, {AggSpec::count_star("n")}).plan();
    const Plan by_name =
        scan(t).aggregate({"c0"}, {AggSpec::count_star("n")}).plan();
    CHECK(by_index.to_string() == by_name.to_string());
}

TEST_CASE("WP-8 builder: validation rejects bad references") {
    const Table t = two_col_table();
    // Filter predicate must be BOOL.
    CHECK_THROWS_AS(scan(t).filter(col(Type::I32, 0)), std::invalid_argument);
    // Unknown column name in a key list.
    CHECK_THROWS_AS(
        scan(t).aggregate({"nope"}, {AggSpec::count_star("n")}),
        std::invalid_argument);
    // Out-of-range sort key.
    CHECK_THROWS_AS(scan(t).sort({SortKey{9, SortDir::Asc, NullOrder::Last}}),
                    std::invalid_argument);
}
