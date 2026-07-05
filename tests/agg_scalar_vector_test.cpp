//  WP-5 GATE: scalar == vector for the masked-reduction kernels (RIGOR.md rule 3
//  / D17). Two layers:
//    1. DIRECT kernel diff over seeded random arrays AND the SIMD-tail boundary
//       lengths (the off-by-a-lane hotspot). Integer reductions compare EXACT;
//       the F64 SUM compares within the D11 epsilon, because horizontal float
//       summation order differs between the vector (lane-wise) and scalar
//       (sequential) paths — bit-exactness is impossible for the same reason the
//       float-aggregate oracle uses a tolerance (D11). MIN/MAX are exact in both
//       (min/max introduces no rounding).
//    2. END-TO-END: a GLOBAL aggregate run via AggKernelPath::kVector vs kScalar
//       must produce IDENTICAL result sets (engine drains both; the global path
//       is the one that drives these kernels in production).
//  Replay any failure:  ./agg_scalar_vector_test --seed N

#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <limits>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "ops/agg_kernels.h"
#include "ops/aggregate.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/result_set.h"
#include "tests/ops_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::ops;

namespace {

const std::vector<std::size_t> kLens = {0,  1,  2,   7,    8,    15,  16,
                                        17, 31, 32,  33,   63,   64,  65,
                                        100, 127, 128, 129, 1000, 2047, 2048};

bool approx(double a, double b) {
    const double diff = std::fabs(a - b);
    return diff <= 1e-9 + 1e-9 * std::max(std::fabs(a), std::fabs(b));
}

}  // namespace

TEST_CASE("scalar==vector: agg_sum_i64 (exact) over random + boundary lengths") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xA661u);
    std::uniform_int_distribution<std::int64_t> dist(-1'000'000'000LL,
                                                     1'000'000'000LL);
    for (std::size_t n : kLens) {
        std::vector<std::int64_t> v(n);
        for (auto& x : v) x = dist(rng);
        CHECK(agg_sum_i64_vec(v.data(), n) == agg_sum_i64_scalar(v.data(), n));
    }
}

TEST_CASE("scalar==vector: agg_sum_f64 (within D11 eps) over boundary lengths") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xF64Du);
    std::uniform_real_distribution<double> dist(-1'000'000.0, 1'000'000.0);
    for (std::size_t n : kLens) {
        std::vector<double> v(n);
        for (auto& x : v) x = dist(rng);
        CHECK(approx(agg_sum_f64_vec(v.data(), n),
                     agg_sum_f64_scalar(v.data(), n)));
    }
}

TEST_CASE("scalar==vector: agg_min/max i64 + f64 (exact), n>=1") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x312Eu);
    std::uniform_int_distribution<std::int64_t> di(-5'000'000LL, 5'000'000LL);
    std::uniform_real_distribution<double> df(-1e6, 1e6);
    for (std::size_t n : kLens) {
        if (n == 0) continue;  // min/max require n>=1
        std::vector<std::int64_t> vi(n);
        std::vector<double> vf(n);
        for (std::size_t k = 0; k < n; ++k) {
            vi[k] = di(rng);
            vf[k] = df(rng);
        }
        CHECK(agg_min_i64_vec(vi.data(), n) == agg_min_i64_scalar(vi.data(), n));
        CHECK(agg_max_i64_vec(vi.data(), n) == agg_max_i64_scalar(vi.data(), n));
        CHECK(agg_min_f64_vec(vf.data(), n) == agg_min_f64_scalar(vf.data(), n));
        CHECK(agg_max_f64_vec(vf.data(), n) == agg_max_f64_scalar(vf.data(), n));
    }
}

// Audit C2: NaN inputs. The F64 MIN/MAX contract is the NaN-greatest TOTAL order
// (MIN == NaN iff ALL elements are NaN; MAX == NaN iff ANY element is). Raw
// hn::Min/Max NaN behavior is ISA-dependent (NEON returns NaN, x86 the second
// operand), so this case is exactly where scalar==vector proves the kernels are
// deterministic across targets. NaN==NaN compared via bit-class, not ==.
TEST_CASE("scalar==vector: agg_min/max f64 with NaN lanes (total order)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x4A4Eu);
    std::uniform_real_distribution<double> df(-1e6, 1e6);
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    auto same = [](double a, double b) {
        return (std::isnan(a) && std::isnan(b)) || a == b;
    };
    for (std::size_t n : kLens) {
        if (n == 0) continue;
        for (int density = 0; density < 3; ++density) {  // some / most / all NaN
            std::vector<double> vf(n);
            for (std::size_t k = 0; k < n; ++k) {
                const bool nan_here = density == 2 || (rng() % 4) < (density ? 3u : 1u);
                vf[k] = nan_here ? qnan : df(rng);
            }
            const double mnv = agg_min_f64_vec(vf.data(), n);
            const double mns = agg_min_f64_scalar(vf.data(), n);
            const double mxv = agg_max_f64_vec(vf.data(), n);
            const double mxs = agg_max_f64_scalar(vf.data(), n);
            CHECK_MESSAGE(same(mnv, mns), "min vec=" << mnv << " scalar=" << mns
                                                     << " n=" << n);
            CHECK_MESSAGE(same(mxv, mxs), "max vec=" << mxv << " scalar=" << mxs
                                                     << " n=" << n);
            // MAX must be NaN iff any lane was NaN (both paths).
            const bool any_nan = std::any_of(vf.begin(), vf.end(),
                                             [](double x) { return std::isnan(x); });
            CHECK(std::isnan(mxs) == any_nan);
            // MIN must be NaN iff every lane was NaN.
            const bool all_nan = std::all_of(vf.begin(), vf.end(),
                                             [](double x) { return std::isnan(x); });
            CHECK(std::isnan(mns) == all_nan);
        }
    }
}

namespace {

// Build a small table (i32 with nulls, f64 with nulls) and a global aggregate.
qe::Table make_table() {
    using namespace qe::ops_test;
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<std::int32_t> a;
    std::vector<double> b;
    std::vector<std::size_t> anull, bnull;
    for (int i = 0; i < 1500; ++i) {
        a.push_back(i - 700);
        b.push_back(static_cast<double>(i) * 0.5 - 100.0);
        if (i % 7 == 0) anull.push_back(static_cast<std::size_t>(i));
        if (i % 13 == 0) bnull.push_back(static_cast<std::size_t>(i));
    }
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(a, anull));
    cols.push_back(f64_col(b, bnull));
    return Table(s, std::move(cols));
}

std::vector<AggSpec> global_aggs() {
    std::vector<AggSpec> v;
    v.push_back(AggSpec::count_star("cs"));
    v.push_back(AggSpec::count(0, "cnt0"));
    v.push_back(AggSpec::sum(0, "sum0"));
    v.push_back(AggSpec::min(0, "min0"));
    v.push_back(AggSpec::max(0, "max0"));
    v.push_back(AggSpec::avg(0, "avg0"));
    v.push_back(AggSpec::sum(1, "sum1"));
    v.push_back(AggSpec::avg(1, "avg1"));
    return v;
}

qe::oracle::ResultSet run_global(const Table& t, AggKernelPath path) {
    auto scan = std::make_unique<Scan>(t, 256);  // small batch => many merges
    auto agg = std::make_unique<Aggregate>(std::move(scan),
                                           std::vector<std::uint32_t>{},
                                           global_aggs());
    agg->set_paths(HashPath::kVector, path);
    return qe::oracle::drain_operator(*agg);
}

}  // namespace

TEST_CASE("end-to-end: global aggregate identical for kVector and kScalar") {
    const Table t = make_table();
    const qe::oracle::ResultSet v = run_global(t, AggKernelPath::kVector);
    const qe::oracle::ResultSet s = run_global(t, AggKernelPath::kScalar);
    const qe::oracle::DiffResult d = qe::oracle::compare_result_sets(v, s);
    CHECK(d.equal);
    if (!d.equal) MESSAGE("kVector vs kScalar diff: " << d.message);
}
