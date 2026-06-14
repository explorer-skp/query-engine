//  WP-2 GATE: seeded property/fuzz over random expression TREES across random
//  batches (with nulls AND selection vectors), at boundary lengths. For each
//  tree the Vector and Scalar whole-tree evaluations must agree on BOTH values
//  and validity. Replay a failure with:  ./expr_property_test --seed N

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "tests/expr_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;

namespace {

// Fixed column layout for generated trees.
//   0:I32  1:I64  2:F64  3:BOOL  4:TS
constexpr std::uint32_t kI32 = 0, kI64 = 1, kF64 = 2, kBOOL = 3, kTS = 4;

Type rand_numeric_type(std::mt19937_64& rng) {
    switch (rng() % 3) {
        case 0: return Type::I32;
        case 1: return Type::I64;
        default: return Type::F64;
    }
}

Expr rand_numeric_lit(std::mt19937_64& rng) {
    switch (rng() % 3) {
        case 0: return lit(Scalar::i32(static_cast<std::int32_t>(rng())));
        case 1: return lit(Scalar::i64(static_cast<std::int64_t>(rng())));
        default:
            return lit(Scalar::f64(
                static_cast<double>(static_cast<std::int32_t>(rng())) / 16.0));
    }
}

Expr gen_bool(std::mt19937_64& rng, int depth);

Expr gen_numeric(std::mt19937_64& rng, int depth) {
    if (depth <= 0 || rng() % 3 == 0) {
        switch (rng() % 6) {
            case 0: return col(Type::I32, kI32);
            case 1: return col(Type::I64, kI64);
            case 2: return col(Type::F64, kF64);
            case 3: return rand_numeric_lit(rng);
            case 4: return cast(col(Type::BOOL, kBOOL), rand_numeric_type(rng));
            default: return cast(col(Type::TS, kTS), rand_numeric_type(rng));
        }
    }
    switch (rng() % 4) {
        case 0: {
            const ArithOp ops[] = {ArithOp::Add, ArithOp::Sub, ArithOp::Mul,
                                   ArithOp::Div, ArithOp::Mod};
            return arith(ops[rng() % 5], gen_numeric(rng, depth - 1),
                         gen_numeric(rng, depth - 1));
        }
        case 1:
            return cast(gen_numeric(rng, depth - 1), rand_numeric_type(rng));
        case 2:
            return cast(gen_bool(rng, depth - 1), rand_numeric_type(rng));
        default:
            return gen_numeric(rng, 0);
    }
}

Expr gen_bool(std::mt19937_64& rng, int depth) {
    const CmpOp cops[] = {CmpOp::Lt, CmpOp::Le, CmpOp::Gt,
                          CmpOp::Ge, CmpOp::Eq, CmpOp::Ne};
    if (depth <= 0 || rng() % 3 == 0) {
        switch (rng() % 3) {
            case 0: return col(Type::BOOL, kBOOL);
            case 1: return lit(Scalar::boolean((rng() & 1u) != 0));
            default:
                return cmp(cops[rng() % 6], gen_numeric(rng, 0),
                           gen_numeric(rng, 0));
        }
    }
    switch (rng() % 4) {
        case 0:
            return logic_and(gen_bool(rng, depth - 1), gen_bool(rng, depth - 1));
        case 1:
            return logic_or(gen_bool(rng, depth - 1), gen_bool(rng, depth - 1));
        case 2:
            return logic_not(gen_bool(rng, depth - 1));
        default:
            return cmp(cops[rng() % 6], gen_numeric(rng, depth - 1),
                       gen_numeric(rng, depth - 1));
    }
}

OwnedBatch build_batch(std::mt19937_64& rng, std::size_t n) {
    std::vector<OwnedColumn> cols;
    cols.push_back(qe::expr::test::random_column(rng, Type::I32, n));
    cols.push_back(qe::expr::test::random_column(rng, Type::I64, n));
    cols.push_back(qe::expr::test::random_column(rng, Type::F64, n));
    cols.push_back(qe::expr::test::random_column(rng, Type::BOOL, n));
    cols.push_back(qe::expr::test::random_column(rng, Type::TS, n));
    return qe::expr::test::make_batch(std::move(cols));
}

}  // namespace

TEST_CASE("property: random trees, scalar==vector on values AND validity") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xF0FFEEu);
    const std::vector<std::size_t> lens = {1, 7, 63, 64, 65, 200, 2047, 2048};

    for (int iter = 0; iter < 400; ++iter) {
        const std::size_t n = lens[rng() % lens.size()];
        OwnedBatch batch = build_batch(rng, n);

        // Half the iterations install a random selection vector (subset/reorder).
        if (n > 0 && (rng() & 1u)) {
            const std::size_t m = 1 + rng() % n;
            std::vector<std::uint32_t> sel(m);
            for (auto& s : sel) s = static_cast<std::uint32_t>(rng() % n);
            batch.set_selection(std::move(sel));
        }
        const Batch bv = batch.view();

        // Root is bool half the time, numeric the other half.
        Expr e = (rng() & 1u) ? gen_bool(rng, 3) : gen_numeric(rng, 3);

        OwnedColumn rv = evaluate(e, bv, Backend::Vector);
        OwnedColumn rs = evaluate(e, bv, Backend::Scalar);
        const bool ok = qe::expr::test::col_equal(rv, rs);
        CHECK(ok);
        if (!ok) {
            MESSAGE("mismatch at iter=" << iter << " n=" << n
                                        << " type=" << static_cast<int>(e.type()));
            break;  // first failure is enough; replay with --seed
        }
    }
}
