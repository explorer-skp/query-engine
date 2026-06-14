//  WP-7 GATE: the two cross-checks that make the dual sort path PROVE something.
//
//   1. RADIX == COMPARISON. The radix fast-path and the comparison path are two
//      INDEPENDENT sort algorithms (a byte-wise LSD radix over order-preserving
//      keys vs std::stable_sort under a tuple comparator). Driven on identical
//      all-integer inputs with random ASC/DESC + NULLS FIRST/LAST, they must
//      produce IDENTICAL sorted result sets (positional compare). A radix
//      sign-bit / null-byte slip shows up here as a disagreement.
//
//   2. GATHER scalar == vector (RIGOR.md rule 3 / D17). The whole-row permutation
//      gather is the SIMD technique of this WP. (a) DIRECT: gather32/gather64
//      vec vs scalar over seeded random data AND the SIMD-tail boundary lengths.
//      (b) END-TO-END: the same sort via GatherPath::kVector vs kScalar yields
//      identical result sets.
//
//  Replay any failure:  ./sort_scalar_vector_test --seed N

#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "oracle/result_set.h"
#include "simd/gather_kernels.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::oracle;

namespace {

const std::vector<std::size_t> kLens = {0,   1,   2,   7,    8,    15,  16,
                                        17,  31,  32,  33,   63,   64,  65,
                                        100, 127, 128, 129,  1000, 2047, 2048};

// An all-integer table (I32, I64, TS) of `n` rows with ~null_pct% nulls, so the
// radix fast-path is eligible on any key subset.
Table make_int_table(std::mt19937_64& rng, std::size_t n, int null_pct = 18) {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    s.fields.emplace_back("c2", Type::TS);
    std::uniform_int_distribution<std::int32_t> d32(-50, 50);  // many ties
    std::uniform_int_distribution<std::int64_t> d64(-1'000'000, 1'000'000);
    std::vector<OwnedColumn> cols;
    {
        OwnedColumn c = OwnedColumn::make(Type::I32, n);
        auto* d = reinterpret_cast<std::int32_t*>(c.mutable_data());
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = d32(rng);
            if (static_cast<int>(rng() % 100) < null_pct) c.set_null(i);
        }
        cols.push_back(std::move(c));
    }
    for (int col = 1; col < 3; ++col) {
        OwnedColumn c = OwnedColumn::make(col == 1 ? Type::I64 : Type::TS, n);
        auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = d64(rng);
            if (static_cast<int>(rng() % 100) < null_pct) c.set_null(i);
        }
        cols.push_back(std::move(c));
    }
    return Table(s, std::move(cols));
}

ResultSet sort_via(const Table& t, const std::vector<SortKey>& keys,
                   Sort::Path path, Sort::GatherPath gp, std::size_t bs) {
    auto scan = std::make_unique<Scan>(t, bs);
    auto sort = std::make_unique<Sort>(std::move(scan), keys);
    sort->set_path(path);
    sort->set_gather_path(gp);
    return drain_operator(*sort);
}

// A random ORDER BY over all three integer columns with random dir + null order.
std::vector<SortKey> rand_int_keys(std::mt19937_64& rng) {
    std::vector<SortKey> keys;
    std::vector<std::uint32_t> cols = {0, 1, 2};
    for (std::uint32_t i = 3; i > 1; --i) std::swap(cols[i - 1], cols[rng() % i]);
    for (std::uint32_t c : cols) {
        SortKey k;
        k.col = c;
        k.dir = (rng() & 1u) ? SortDir::Desc : SortDir::Asc;
        k.nulls = (rng() & 1u) ? NullOrder::First : NullOrder::Last;
        keys.push_back(k);
    }
    return keys;
}

}  // namespace

TEST_CASE("scalar==vector: gather32/gather64 direct over boundary lengths") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5031Au);
    for (std::size_t n : kLens) {
        // Source big enough for any index; indices in [0, src_len).
        const std::size_t src_len = n + 1;
        std::vector<std::uint32_t> src32(src_len), idx(n);
        std::vector<std::uint64_t> src64(src_len);
        for (std::size_t i = 0; i < src_len; ++i) {
            src32[i] = static_cast<std::uint32_t>(rng());
            src64[i] = rng();
        }
        for (std::size_t k = 0; k < n; ++k)
            idx[k] = static_cast<std::uint32_t>(rng() % src_len);

        std::vector<std::uint32_t> ov32(n), os32(n);
        std::vector<std::uint64_t> ov64(n), os64(n);
        simd::gather32_vec(src32.data(), idx.data(), n, ov32.data());
        simd::gather32_scalar(src32.data(), idx.data(), n, os32.data());
        simd::gather64_vec(src64.data(), idx.data(), n, ov64.data());
        simd::gather64_scalar(src64.data(), idx.data(), n, os64.data());
        CHECK(ov32 == os32);
        CHECK(ov64 == os64);
    }
}

TEST_CASE("radix == comparison on identical all-integer inputs") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xAD1C5A11u);
    for (int iter = 0; iter < 60; ++iter) {
        const std::size_t n = rng() % 5000;
        const Table t = make_int_table(rng, n);
        const std::vector<SortKey> keys = rand_int_keys(rng);
        const std::size_t bs = (rng() % 2) ? 64 : 2048;

        const ResultSet r = sort_via(t, keys, Sort::Path::kRadix,
                                     Sort::GatherPath::kVector, bs);
        const ResultSet c = sort_via(t, keys, Sort::Path::kComparison,
                                     Sort::GatherPath::kVector, bs);
        const DiffResult d = compare_result_sets(r, c, /*ordered=*/true);
        CHECK(d.equal);
        if (!d.equal) {
            MESSAGE("radix!=comparison iter=" << iter << " n=" << n << " : "
                                              << d.message);
            break;
        }
    }
}

TEST_CASE("end-to-end: sorted result identical for kVector and kScalar gather") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6A746Eu);
    for (int iter = 0; iter < 40; ++iter) {
        const std::size_t n = rng() % 5000;
        const Table t = make_int_table(rng, n);
        const std::vector<SortKey> keys = rand_int_keys(rng);
        const ResultSet v = sort_via(t, keys, Sort::Path::kComparison,
                                     Sort::GatherPath::kVector, 256);
        const ResultSet s = sort_via(t, keys, Sort::Path::kComparison,
                                     Sort::GatherPath::kScalar, 256);
        const DiffResult d = compare_result_sets(v, s, /*ordered=*/true);
        CHECK(d.equal);
        if (!d.equal) MESSAGE("gather kVector!=kScalar: " << d.message);
    }
}
