//  WP-2 GATE: scalar == vector for EVERY op×type kernel, over seeded random
//  inputs including boundary lengths (the SIMD tail: 63/64/65/127/2047/2048).
//  This is the differential that makes the mutation self-test meaningful.
//  Replay any failure with:  ./expr_scalar_vector_test --seed N

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "expr/kernels.h"
#include "tests/expr_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;

namespace {
const std::vector<std::size_t> kLens = {0,  1,   2,   7,    15,   31,  32,
                                        33, 63,  64,  65,   100,  127, 128,
                                        129, 255, 256, 257, 1000, 2047, 2048};

template <typename T>
std::vector<T> rand_vec(std::mt19937_64& rng, std::size_t n) {
    std::vector<T> v(n);
    for (auto& x : v) x = static_cast<T>(rng());
    return v;
}
}  // namespace

TEST_CASE("scalar==vector: arithmetic, all ops × {I32,I64,F64}, boundary lens") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xA1Du);
    const ArithOp ops[] = {ArithOp::Add, ArithOp::Sub, ArithOp::Mul,
                           ArithOp::Div, ArithOp::Mod};
    for (std::size_t n : kLens) {
        for (ArithOp op : ops) {
            // ---- I32 ----
            {
                auto a = rand_vec<std::int32_t>(rng, n);
                auto b = rand_vec<std::int32_t>(rng, n);
                // sprinkle 0 and -1 divisors to hit the guarded paths
                for (std::size_t i = 0; i < n; i += 5) b[i] = 0;
                for (std::size_t i = 2; i < n; i += 7) b[i] = -1;
                if (n) a[0] = INT32_MIN;  // INT_MIN/-1 overflow case
                std::vector<std::int32_t> ov(n), os(n);
                arith_vec(op, Type::I32, a.data(), b.data(), ov.data(), n);
                arith_scalar(op, Type::I32, a.data(), b.data(), os.data(), n);
                CHECK(ov == os);
            }
            // ---- I64 ----
            {
                auto a = rand_vec<std::int64_t>(rng, n);
                auto b = rand_vec<std::int64_t>(rng, n);
                for (std::size_t i = 0; i < n; i += 5) b[i] = 0;
                for (std::size_t i = 2; i < n; i += 7) b[i] = -1;
                if (n) a[0] = INT64_MIN;
                std::vector<std::int64_t> ov(n), os(n);
                arith_vec(op, Type::I64, a.data(), b.data(), ov.data(), n);
                arith_scalar(op, Type::I64, a.data(), b.data(), os.data(), n);
                CHECK(ov == os);
            }
            // ---- F64 ----
            {
                std::vector<double> a(n), b(n);
                for (std::size_t i = 0; i < n; ++i) {
                    a[i] = static_cast<double>(static_cast<std::int64_t>(rng())) /
                           1024.0;
                    b[i] = static_cast<double>(static_cast<std::int64_t>(rng())) /
                           1024.0;
                }
                for (std::size_t i = 0; i < n; i += 6) b[i] = 0.0;  // /0 => inf/nan
                std::vector<double> ov(n), os(n);
                arith_vec(op, Type::F64, a.data(), b.data(), ov.data(), n);
                arith_scalar(op, Type::F64, a.data(), b.data(), os.data(), n);
                for (std::size_t i = 0; i < n; ++i)
                    CHECK(qe::expr::test::double_eq(ov[i], os[i]));
            }
        }
    }
}

TEST_CASE("scalar==vector: comparison, all ops × all types, boundary lens") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xC03u);
    const CmpOp ops[] = {CmpOp::Lt, CmpOp::Le, CmpOp::Gt,
                         CmpOp::Ge, CmpOp::Eq, CmpOp::Ne};
    const Type types[] = {Type::I32, Type::I64, Type::F64, Type::BOOL, Type::TS};
    for (std::size_t n : kLens) {
        for (CmpOp op : ops) {
            for (Type t : types) {
                std::vector<std::uint8_t> a8, b8;
                std::vector<std::int32_t> a32, b32;
                std::vector<std::int64_t> a64, b64;
                std::vector<double> ad, bd;
                const void *pa = nullptr, *pb = nullptr;
                if (t == Type::BOOL) {
                    a8.resize(n); b8.resize(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        a8[i] = rng() & 1u; b8[i] = rng() & 1u;
                    }
                    pa = a8.data(); pb = b8.data();
                } else if (t == Type::I32) {
                    a32 = rand_vec<std::int32_t>(rng, n);
                    b32 = rand_vec<std::int32_t>(rng, n);
                    // force some equal lanes
                    for (std::size_t i = 0; i < n; i += 4) b32[i] = a32[i];
                    pa = a32.data(); pb = b32.data();
                } else if (t == Type::F64) {
                    ad.resize(n); bd.resize(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        ad[i] = static_cast<double>(static_cast<std::int32_t>(rng()));
                        bd[i] = (i % 4 == 0) ? ad[i]
                                             : static_cast<double>(
                                                   static_cast<std::int32_t>(rng()));
                    }
                    pa = ad.data(); pb = bd.data();
                } else {  // I64 or TS
                    a64 = rand_vec<std::int64_t>(rng, n);
                    b64 = rand_vec<std::int64_t>(rng, n);
                    for (std::size_t i = 0; i < n; i += 4) b64[i] = a64[i];
                    pa = a64.data(); pb = b64.data();
                }
                std::vector<std::uint8_t> ov(n), os(n);
                cmp_vec(op, t, pa, pb, ov.data(), n);
                cmp_scalar(op, t, pa, pb, os.data(), n);
                CHECK(ov == os);
            }
        }
    }
}

TEST_CASE("scalar==vector: three-valued logic (tri-state), boundary lens") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x106u);
    for (std::size_t n : kLens) {
        std::vector<std::uint8_t> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<std::uint8_t>(rng() % 3);  // 0/1/2
            b[i] = static_cast<std::uint8_t>(rng() % 3);
        }
        std::vector<std::uint8_t> ov(n), os(n);
        logic_and_vec(a.data(), b.data(), ov.data(), n);
        logic_and_scalar(a.data(), b.data(), os.data(), n);
        CHECK(ov == os);
        logic_or_vec(a.data(), b.data(), ov.data(), n);
        logic_or_scalar(a.data(), b.data(), os.data(), n);
        CHECK(ov == os);
        logic_not_vec(a.data(), ov.data(), n);
        logic_not_scalar(a.data(), os.data(), n);
        CHECK(ov == os);
    }
}

TEST_CASE("REGRESSION (pinned bug): F64->I64/TS cast of large integral & "
          "halfway values — vectorized round-half-away must match std::round") {
    // Pinned bug: the vector round-half-away formula Trunc(x + copysign(0.5,x))
    // corrupts already-integral doubles with |x| >= 2^52 (ULP >= 1, so adding
    // 0.5 rounds to the NEXT integer). Discovered via expr_property_test
    // --seed 14745959599513406255 (iter 199, F64->I64). Scalar twin (std::round)
    // is correct; this asserts the vector kernel agrees, value-for-value.
    std::vector<double> xs = {
        -7901924385117227.0,  // the exact value that first exposed the bug
        7901924385117227.0,
        4503599627370496.0,        // 2^52
        4503599627370497.0,        // 2^52 + 1
        -4503599627370496.0,       // -2^52
        9007199254740991.0,        // 2^53 - 1 (largest exact odd integer * ...)
        -9007199254740991.0,
        123456789012345.0,
        -123456789012345.0,
        2.5,  -2.5,  3.5,  -3.5,  0.5,  -0.5,  100000.5,  -100000.5,
        2.4,  -2.4,  2.6,  -2.6,  0.0,  -0.0,
    };
    // Plus a sweep of large-magnitude INTEGRAL doubles (|x| in [2^52, 2^53)),
    // which must round to themselves.
    std::mt19937_64 rng(qe::test::seed() ^ 0xB16u);
    for (int i = 0; i < 5000; ++i) {
        const double sign = (rng() & 1u) ? 1.0 : -1.0;
        const std::uint64_t k =
            (1ull << 52) + (rng() % ((1ull << 52)));  // integral, in [2^52,2^53)
        xs.push_back(sign * static_cast<double>(k));
    }
    const std::size_t n = xs.size();
    for (Type to : {Type::I64, Type::TS}) {
        std::vector<std::int64_t> ov(n), os(n);
        cast_vec(Type::F64, to, xs.data(), ov.data(), n);
        cast_scalar(Type::F64, to, xs.data(), os.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(ov[i] == os[i]);
            // For integral inputs the result is exactly the integer.
            if (xs[i] == std::trunc(xs[i]) && std::abs(xs[i]) < 9.0e18)
                CHECK(ov[i] == static_cast<std::int64_t>(xs[i]));
        }
    }
}

TEST_CASE("scalar==vector: cast, every meaningful (from,to) pair, boundary lens") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xCA57u);
    const Type types[] = {Type::I32, Type::I64, Type::F64, Type::BOOL, Type::TS};
    for (std::size_t n : kLens) {
        for (Type from : types) {
            for (Type to : types) {
                if (from == to) continue;
                // Build source bytes of `from`.
                std::vector<std::byte> in(n * byte_width(from));
                void* ip = in.data();
                for (std::size_t i = 0; i < n; ++i) {
                    switch (from) {
                        case Type::I32:
                            static_cast<std::int32_t*>(ip)[i] =
                                static_cast<std::int32_t>(rng());
                            break;
                        case Type::I64:
                        case Type::TS:
                            static_cast<std::int64_t*>(ip)[i] =
                                static_cast<std::int64_t>(rng());
                            break;
                        case Type::STR:
                            break;  // WP-7b: STR excluded from the cast type matrix
                        case Type::F64: {
                            // integer-valued + simple half fractions + some OOR,
                            // avoiding ULP-half pathologies so round-half-away is
                            // identical in both paths.
                            const std::int64_t base =
                                static_cast<std::int64_t>(rng()) % 100000;
                            const double frac[] = {0.0,  0.25, 0.5,
                                                   0.75, -0.5, -0.25};
                            double v = static_cast<double>(base) +
                                       frac[rng() % 6];
                            if (rng() % 20 == 0) v = 1e300;  // out of int range
                            if (rng() % 20 == 1) v = -1e300;
                            static_cast<double*>(ip)[i] = v;
                            break;
                        }
                        case Type::BOOL:
                            static_cast<std::uint8_t*>(ip)[i] =
                                (rng() & 1u) ? 1 : 0;
                            break;
                    }
                }
                std::vector<std::byte> ov(n * byte_width(to));
                std::vector<std::byte> os(n * byte_width(to));
                cast_vec(from, to, in.data(), ov.data(), n);
                cast_scalar(from, to, in.data(), os.data(), n);
                // Compare produced VALUES bit-for-bit (both paths write a defined
                // value at every lane).
                CHECK(ov == os);
            }
        }
    }
}
