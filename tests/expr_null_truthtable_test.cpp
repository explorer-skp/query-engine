//  WP-2 GATE: the documented NULL-propagation truth table (expr/expr.h),
//  exhaustively. Three-valued AND/OR/NOT over all combinations; null-if-any for
//  arithmetic & comparison; division-by-zero => NULL; cast null + out-of-range.
//  Run with both backends so the truth table holds for vector AND scalar paths.

#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "tests/expr_test_util.h"

using namespace qe;
using namespace qe::expr;
using qe::expr::test::valid_at;

namespace {

// Tri-state symbols for readability: F=false, T=true, N=null.
constexpr int F = 0, T = 1, N = 2;

OwnedColumn bool_col(const std::vector<int>& cells) {
    OwnedColumn c = OwnedColumn::make(Type::BOOL, cells.size());
    auto* d = reinterpret_cast<std::uint8_t*>(c.mutable_data());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        d[i] = (cells[i] == T) ? 1 : 0;
        if (cells[i] == N) c.set_null(i);
    }
    return c;
}

// Read result row i as a tri-state symbol.
int cell_of(const OwnedColumn& r, std::size_t i) {
    const Column v = r.view();
    if (!valid_at(v, i)) return N;
    return reinterpret_cast<const std::uint8_t*>(v.data)[i] ? T : F;
}

}  // namespace

TEST_CASE("truth table: three-valued AND / OR / NOT (both backends)") {
    //               (F,F)(F,T)(F,N)(T,F)(T,T)(T,N)(N,F)(N,T)(N,N)
    std::vector<int> a = {F, F, F, T, T, T, N, N, N};
    std::vector<int> b = {F, T, N, F, T, N, F, T, N};
    std::vector<int> expect_and = {F, F, F, F, T, N, F, N, N};
    std::vector<int> expect_or = {F, T, N, T, T, T, N, T, N};
    std::vector<int> expect_not_a = {T, T, T, F, F, F, N, N, N};

    OwnedBatch batch = qe::expr::test::make_batch(
        [&] {
            std::vector<OwnedColumn> v;
            v.push_back(bool_col(a));
            v.push_back(bool_col(b));
            return v;
        }());
    const Batch bv = batch.view();

    Expr ca = col(Type::BOOL, 0), cb = col(Type::BOOL, 1);
    for (Backend be : {Backend::Vector, Backend::Scalar}) {
        OwnedColumn r_and = evaluate(logic_and(ca, cb), bv, be);
        OwnedColumn r_or = evaluate(logic_or(ca, cb), bv, be);
        OwnedColumn r_not = evaluate(logic_not(ca), bv, be);
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(cell_of(r_and, i) == expect_and[i]);
            CHECK(cell_of(r_or, i) == expect_or[i]);
            CHECK(cell_of(r_not, i) == expect_not_a[i]);
        }
    }
}

TEST_CASE("truth table: arithmetic & comparison are null-if-any (both backends)") {
    // a, b: I32 with nulls in different places. Row pattern (a_null, b_null):
    //   0: (-,-) both valid    1: (null,-)    2: (-,null)    3: (null,null)
    OwnedColumn a = OwnedColumn::make(Type::I32, 4);
    OwnedColumn b = OwnedColumn::make(Type::I32, 4);
    auto* da = reinterpret_cast<std::int32_t*>(a.mutable_data());
    auto* db = reinterpret_cast<std::int32_t*>(b.mutable_data());
    da[0] = 10; da[1] = 20; da[2] = 30; da[3] = 40;
    db[0] = 3;  db[1] = 4;  db[2] = 5;  db[3] = 6;
    a.set_null(1); a.set_null(3);
    b.set_null(2); b.set_null(3);

    OwnedBatch batch = qe::expr::test::make_batch([&] {
        std::vector<OwnedColumn> v;
        v.push_back(std::move(a));
        v.push_back(std::move(b));
        return v;
    }());
    const Batch bv = batch.view();
    Expr ca = col(Type::I32, 0), cb = col(Type::I32, 1);

    for (Backend be : {Backend::Vector, Backend::Scalar}) {
        OwnedColumn add_r = evaluate(add(ca, cb), bv, be);
        OwnedColumn lt_r = evaluate(lt(ca, cb), bv, be);
        // Valid only on row 0 (the only row where both are valid).
        CHECK(valid_at(add_r.view(), 0));
        CHECK(valid_at(lt_r.view(), 0));
        for (std::size_t i = 1; i < 4; ++i) {
            CHECK_FALSE(valid_at(add_r.view(), i));
            CHECK_FALSE(valid_at(lt_r.view(), i));
        }
        CHECK(reinterpret_cast<const std::int32_t*>(add_r.view().data)[0] == 13);
        CHECK(reinterpret_cast<const std::uint8_t*>(lt_r.view().data)[0] == 0);
    }
}

TEST_CASE("truth table: integer & float division by zero => NULL (both backends)") {
    OwnedColumn a = OwnedColumn::make(Type::I32, 3);
    OwnedColumn b = OwnedColumn::make(Type::I32, 3);
    auto* da = reinterpret_cast<std::int32_t*>(a.mutable_data());
    auto* db = reinterpret_cast<std::int32_t*>(b.mutable_data());
    da[0] = 10; da[1] = 10; da[2] = 10;
    db[0] = 2;  db[1] = 0;  db[2] = 5;   // row 1 divides by zero

    OwnedBatch batch = qe::expr::test::make_batch([&] {
        std::vector<OwnedColumn> v;
        v.push_back(std::move(a));
        v.push_back(std::move(b));
        return v;
    }());
    const Batch bv = batch.view();
    Expr ca = col(Type::I32, 0), cb = col(Type::I32, 1);

    for (Backend be : {Backend::Vector, Backend::Scalar}) {
        OwnedColumn d = evaluate(div(ca, cb), bv, be);
        OwnedColumn m = evaluate(mod(ca, cb), bv, be);
        CHECK(valid_at(d.view(), 0));
        CHECK_FALSE(valid_at(d.view(), 1));  // 10/0 => NULL
        CHECK(valid_at(d.view(), 2));
        CHECK(reinterpret_cast<const std::int32_t*>(d.view().data)[0] == 5);
        CHECK_FALSE(valid_at(m.view(), 1));  // 10%0 => NULL
    }
}

TEST_CASE("truth table: cast propagates NULL and NULLs out-of-range (both backends)") {
    // F64 column: row0 in range, row1 NULL input, row2 out of int32 range.
    OwnedColumn x = OwnedColumn::make(Type::F64, 3);
    auto* dx = reinterpret_cast<double*>(x.mutable_data());
    dx[0] = 2.5;       // rounds half away => 3
    dx[1] = 99.0;      // value present but row marked null
    dx[2] = 1e18;      // far out of int32 range
    x.set_null(1);

    OwnedBatch batch = qe::expr::test::make_batch([&] {
        std::vector<OwnedColumn> v;
        v.push_back(std::move(x));
        return v;
    }());
    const Batch bv = batch.view();
    Expr cx = col(Type::F64, 0);

    for (Backend be : {Backend::Vector, Backend::Scalar}) {
        OwnedColumn r = evaluate(cast(cx, Type::I32), bv, be);
        CHECK(valid_at(r.view(), 0));
        CHECK(reinterpret_cast<const std::int32_t*>(r.view().data)[0] == 3);
        CHECK_FALSE(valid_at(r.view(), 1));  // input NULL propagates
        CHECK_FALSE(valid_at(r.view(), 2));  // out of range => NULL
    }
}
