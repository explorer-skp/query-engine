//  WP-1: selection-vector semantics + gather kernels.
//   * sel == nullptr is the dense identity; sel != nullptr selects/reorders;
//   * gather scalar == vector over randomized inputs (seeded);
//   * compaction is correct (data + nulls) and out-of-place (no §12 aliasing).

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/selection.h"
#include "core/types.h"
#include "simd/gather_kernels.h"
#include "tests/wp1_seed.h"

using namespace qe;

TEST_CASE("selection helpers: dense vs selected") {
    CHECK(sel_at(nullptr, 0) == 0u);
    CHECK(sel_at(nullptr, 42) == 42u);
    CHECK(selected_count(nullptr, 100) == 100u);

    std::vector<std::uint32_t> idx = {5, 2, 9, 0};
    SelectionVector sv{idx.data(), idx.size()};
    CHECK(sel_at(&sv, 0) == 5u);
    CHECK(sel_at(&sv, 2) == 9u);
    CHECK(selected_count(&sv, 100) == 4u);
}

TEST_CASE("gather32 / gather64: scalar == vector over random sel") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6A7u);
    for (int iter = 0; iter < 2000; ++iter) {
        const std::size_t src_n = 1 + rng() % 1000;
        const std::size_t n = rng() % src_n + 1;
        std::vector<std::uint32_t> idx(n);
        for (auto& x : idx) x = static_cast<std::uint32_t>(rng() % src_n);

        std::vector<std::uint32_t> s32(src_n);
        std::vector<std::uint64_t> s64(src_n);
        for (std::size_t i = 0; i < src_n; ++i) {
            s32[i] = static_cast<std::uint32_t>(rng());
            s64[i] = rng();
        }

        std::vector<std::uint32_t> o32s(n), o32v(n);
        simd::gather32_scalar(s32.data(), idx.data(), n, o32s.data());
        simd::gather32_vec(s32.data(), idx.data(), n, o32v.data());
        CHECK(o32s == o32v);
        for (std::size_t k = 0; k < n; ++k) CHECK(o32s[k] == s32[idx[k]]);

        std::vector<std::uint64_t> o64s(n), o64v(n);
        simd::gather64_scalar(s64.data(), idx.data(), n, o64s.data());
        simd::gather64_vec(s64.data(), idx.data(), n, o64v.data());
        CHECK(o64s == o64v);
        for (std::size_t k = 0; k < n; ++k) CHECK(o64s[k] == s64[idx[k]]);
    }
}

TEST_CASE("gather dense (idx == nullptr) is the identity copy") {
    std::vector<std::uint32_t> s = {10, 20, 30, 40, 50};
    std::vector<std::uint32_t> ov(5), os(5);
    simd::gather32_vec(s.data(), nullptr, 5, ov.data());
    simd::gather32_scalar(s.data(), nullptr, 5, os.data());
    CHECK(ov == s);
    CHECK(os == s);
}

TEST_CASE("compact_column: correct data + nulls, out-of-place") {
    // Build a 10-value I32 column 0,10,20,...,90, with nulls at rows 2 and 6.
    OwnedColumn in = OwnedColumn::make(Type::I32, 10);
    auto* d = reinterpret_cast<std::int32_t*>(in.mutable_data());
    for (int i = 0; i < 10; ++i) d[i] = i * 10;
    in.set_null(2);
    in.set_null(6);
    Column iv = in.view();

    // Select rows {6, 1, 2, 9} (non-monotonic, includes both null rows 2 & 6).
    std::vector<std::uint32_t> idx = {6, 1, 2, 9};
    SelectionVector sv{idx.data(), idx.size()};

    OwnedColumn out = compact_column(iv, &sv, idx.size());
    Column ov = out.view();

    REQUIRE(ov.len == 4);
    CHECK(out.data() != in.data());  // out-of-place: distinct buffers
    const auto* od = reinterpret_cast<const std::int32_t*>(ov.data);
    CHECK(od[0] == 60);
    CHECK(od[1] == 10);
    CHECK(od[2] == 20);
    CHECK(od[3] == 90);

    // Nulls follow the selection: out rows 0 (was 6) and 2 (was 2) are null.
    CHECK_FALSE(ov.all_valid);
    REQUIRE(ov.validity != nullptr);
    auto bit = [&](std::size_t i) {
        return (ov.validity[i >> 6] >> (i & 63u)) & 1ull;
    };
    CHECK(bit(0) == 0);  // from row 6 (null)
    CHECK(bit(1) == 1);  // from row 1 (valid)
    CHECK(bit(2) == 0);  // from row 2 (null)
    CHECK(bit(3) == 1);  // from row 9 (valid)
}

TEST_CASE("compact_column: all-valid input stays on the fast path") {
    OwnedColumn in = OwnedColumn::make(Type::F64, 8);
    auto* d = reinterpret_cast<double*>(in.mutable_data());
    for (int i = 0; i < 8; ++i) d[i] = i + 0.5;

    std::vector<std::uint32_t> idx = {7, 0, 3};
    SelectionVector sv{idx.data(), idx.size()};
    OwnedColumn out = compact_column(in.view(), &sv, idx.size());
    Column ov = out.view();

    CHECK(ov.all_valid);
    CHECK(ov.validity == nullptr);  // precedence: no nulls => fast path
    const auto* od = reinterpret_cast<const double*>(ov.data);
    CHECK(od[0] == doctest::Approx(7.5));
    CHECK(od[1] == doctest::Approx(0.5));
    CHECK(od[2] == doctest::Approx(3.5));
}

TEST_CASE("compact_column: dense (sel == nullptr) copies identically") {
    OwnedColumn in = OwnedColumn::make(Type::I64, 6);
    auto* d = reinterpret_cast<std::int64_t*>(in.mutable_data());
    for (int i = 0; i < 6; ++i) d[i] = 1000 + i;
    in.set_null(4);

    OwnedColumn out = compact_column(in.view(), nullptr, 6);
    Column ov = out.view();
    const auto* od = reinterpret_cast<const std::int64_t*>(ov.data);
    for (int i = 0; i < 6; ++i) CHECK(od[i] == 1000 + i);
    REQUIRE(ov.validity != nullptr);
    CHECK(((ov.validity[0] >> 4) & 1ull) == 0);  // row 4 still null
}

TEST_CASE("compact_column: BOOL (1-byte) path") {
    OwnedColumn in = OwnedColumn::make(Type::BOOL, 5);
    auto* d = reinterpret_cast<std::uint8_t*>(in.mutable_data());
    d[0] = 1; d[1] = 0; d[2] = 1; d[3] = 1; d[4] = 0;
    std::vector<std::uint32_t> idx = {4, 2, 0};
    SelectionVector sv{idx.data(), idx.size()};
    OwnedColumn out = compact_column(in.view(), &sv, idx.size());
    const auto* od = reinterpret_cast<const std::uint8_t*>(out.view().data);
    CHECK(od[0] == 0);
    CHECK(od[1] == 1);
    CHECK(od[2] == 1);
}
