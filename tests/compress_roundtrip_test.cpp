//  WP-14: PURE round-trip unit test (engine-only, NO DuckDB). For every physical
//  type, decode(encode(col)) is BIT-EXACT on every non-null value and the validity
//  bitmap matches EXACTLY. This is where Gorilla / delta-of-delta / zig-zag / bit-
//  pack correctness across ALL bit patterns is proven — it does NOT go through the
//  divergence-safe oracle grammar, so it exercises the FULL float domain: several
//  NaN payloads, +0.0 AND -0.0 (asserted to round-trip DISTINCTLY), ±inf,
//  subnormals, plus interleaved nulls. Compares the raw 8 bytes / bit pattern (never
//  `==`), so NaN and -0.0 are checked correctly (RIGOR.md float-lossless rule).
//
//  Streaming decode is stressed by sweeping batch sizes INCLUDING non-multiples of
//  64 (1, 7, 13, 64, 256, 2048) — CompressedScan materializes fresh columns each
//  batch, so it has no word-alignment constraint and decode must resume correctly
//  across arbitrary window boundaries.
//
//  Determinism: seed printed (tests/wp1_test_main.cpp). Replay:
//      ./compress_roundtrip_test --seed N

#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "core/validity.h"
#include "ops/table.h"
#include "tests/wp1_seed.h"
#include "tsx/compress.h"

using namespace qe;
using namespace qe::tsx;

namespace {

// (valid, raw bit pattern) of a column cell — the bit-exact comparison unit.
// I32 zero-extends its 32 bits; F64 takes the 8 bytes verbatim (NaN/-0.0 safe).
std::pair<bool, std::uint64_t> cell_bits(const Column& c, std::size_t r) {
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) return {false, 0};
    switch (c.type) {
        case Type::I32:
        case Type::STR:  // WP-7b: not exercised by compress; treat code as 32 bits
            return {true, static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                              reinterpret_cast<const std::int32_t*>(c.data)[r]))};
        case Type::I64:
        case Type::TS:
            return {true, static_cast<std::uint64_t>(
                              reinterpret_cast<const std::int64_t*>(c.data)[r])};
        case Type::F64: {
            std::uint64_t b;
            std::memcpy(&b, reinterpret_cast<const double*>(c.data) + r, 8);
            return {true, b};
        }
        case Type::BOOL:
            return {true, reinterpret_cast<const std::uint8_t*>(c.data)[r] ? 1u
                                                                           : 0u};
    }
    return {false, 0};
}

// Drain a CompressedScan into per-column (valid, bits) sequences of length nrows.
std::vector<std::vector<std::pair<bool, std::uint64_t>>> drain_bits(
    CompressedScan& sc, std::size_t ncol) {
    std::vector<std::vector<std::pair<bool, std::uint64_t>>> out(ncol);
    sc.open();
    while (auto b = sc.next()) {
        const Batch& batch = *b;
        for (std::size_t c = 0; c < ncol; ++c)
            for (std::size_t r = 0; r < batch.row_count; ++r)
                out[c].push_back(cell_bits(batch.cols[c], r));
    }
    sc.close();
    return out;
}

// Source (valid, bits) for one OwnedColumn.
std::vector<std::pair<bool, std::uint64_t>> source_bits(const OwnedColumn& oc) {
    const Column v = oc.view();
    std::vector<std::pair<bool, std::uint64_t>> out;
    out.reserve(oc.len());
    for (std::size_t r = 0; r < oc.len(); ++r) out.push_back(cell_bits(v, r));
    return out;
}

// Assert decode(encode(table)) is bit-exact at the given batch size.
void check_roundtrip(const Table& src, std::size_t bs) {
    const CompressedTable ct = CompressedTable::encode(src);
    REQUIRE(ct.num_rows() == src.num_rows());
    REQUIRE(ct.num_columns() == src.num_columns());
    CompressedScan sc(ct, bs);
    const auto got = drain_bits(sc, src.num_columns());
    for (std::size_t c = 0; c < src.num_columns(); ++c) {
        const auto want = source_bits(src.column(c));
        REQUIRE(got[c].size() == want.size());
        for (std::size_t r = 0; r < want.size(); ++r) {
            CHECK_MESSAGE(got[c][r].first == want[r].first,
                          "validity mismatch col=" << c << " row=" << r);
            CHECK_MESSAGE(got[c][r] == want[r],
                          "bit mismatch col=" << c << " row=" << r
                                              << " bs=" << bs);
        }
    }
}

const std::size_t kBatchSizes[] = {1, 7, 13, 64, 256, 2048};

OwnedColumn make_i32(const std::vector<std::int32_t>& v,
                     const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::I32, v.size());
    auto* d = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}
OwnedColumn make_i64(Type t, const std::vector<std::int64_t>& v,
                     const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(t, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}
OwnedColumn make_f64(const std::vector<double>& v,
                     const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::F64, v.size());
    auto* d = reinterpret_cast<double*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}
OwnedColumn make_bool(const std::vector<std::uint8_t>& v,
                      const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::BOOL, v.size());
    auto* d = reinterpret_cast<std::uint8_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i] ? 1 : 0;
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

Table one_col(Schema s, OwnedColumn c) {
    std::vector<OwnedColumn> cols;
    cols.push_back(std::move(c));
    return Table(std::move(s), std::move(cols));
}
Schema sch1(const char* name, Type t) {
    Schema s;
    s.fields.emplace_back(name, t);
    return s;
}

}  // namespace

TEST_CASE("WP-14 round-trip: F64 Gorilla over the FULL double domain (+ nulls)") {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double snan = std::numeric_limits<double>::signaling_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double denorm_min = std::numeric_limits<double>::denorm_min();
    // A NaN with a custom payload (distinct mantissa bits).
    std::uint64_t nan_payload_bits = 0x7ff8000000000abcULL;
    double nan_payload;
    std::memcpy(&nan_payload, &nan_payload_bits, 8);

    std::vector<double> v = {
        0.0, -0.0, 1.0, -1.0, inf, -inf, qnan, snan, nan_payload,
        denorm_min, -denorm_min, 3.5e-308 /*subnormal-ish*/, 1.7976931348623157e308,
        -2.2250738585072014e-308 /*-min normal*/, 42.0, 42.0, 42.5, 0.1, -0.1, 123456.789};
    Table t = one_col(sch1("f", Type::F64), make_f64(v, /*nulls=*/{2, 9, 15}));
    for (std::size_t bs : kBatchSizes) check_roundtrip(t, bs);

    // Explicit: -0.0 round-trips DISTINCTLY from +0.0.
    const CompressedTable ct = CompressedTable::encode(t);
    CompressedScan sc(ct, 2048);
    const auto got = drain_bits(sc, 1);
    std::uint64_t pos0, neg0;
    std::memcpy(&pos0, &v[0], 8);
    std::memcpy(&neg0, &v[1], 8);
    CHECK(got[0][0].second == pos0);  // +0.0
    CHECK(got[0][1].second == neg0);  // -0.0
    CHECK(pos0 != neg0);              // they really are different bit patterns
}

TEST_CASE("WP-14 round-trip: TS delta-of-delta incl. extreme int64 (+ nulls)") {
    const std::int64_t imax = std::numeric_limits<std::int64_t>::max();
    const std::int64_t imin = std::numeric_limits<std::int64_t>::min();
    std::vector<std::int64_t> v = {0, 1, 2, 100, 99, 1'000'000, imax, imin, 0,
                                   -5, imax - 1, imin + 1, 7, 7, 7};
    Table t = one_col(sch1("t", Type::TS), make_i64(Type::TS, v, {3, 8}));
    for (std::size_t bs : kBatchSizes) check_roundtrip(t, bs);
}

TEST_CASE("WP-14 round-trip: I64 + I32 zig-zag/varint (+ nulls)") {
    std::vector<std::int64_t> v64 = {0, -1, 1, -1000, 1000, 123456789, -987654321};
    Table t64 = one_col(sch1("a", Type::I64), make_i64(Type::I64, v64, {0, 4}));
    std::vector<std::int32_t> v32 = {0, -1, 2147483647, -2147483648, 5, -5, 100};
    Table t32 = one_col(sch1("b", Type::I32), make_i32(v32, {2, 6}));
    for (std::size_t bs : kBatchSizes) {
        check_roundtrip(t64, bs);
        check_roundtrip(t32, bs);
    }
}

TEST_CASE("WP-14 round-trip: BOOL bit-pack (+ nulls); all-null column") {
    std::vector<std::uint8_t> v = {1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
    Table t = one_col(sch1("c", Type::BOOL), make_bool(v, {1, 5, 9}));
    for (std::size_t bs : kBatchSizes) check_roundtrip(t, bs);

    // A column of ALL NULLS compresses to just its bitmap (no value bytes) and
    // still round-trips (every slot stays null).
    OwnedColumn an = OwnedColumn::make(Type::I64, 20);
    for (std::size_t i = 0; i < 20; ++i) an.set_null(i);
    Table allnull = one_col(sch1("d", Type::I64), std::move(an));
    const CompressedTable ct = CompressedTable::encode(allnull);
    CHECK(ct.encoded(0).values.empty());  // no non-null values encoded
    for (std::size_t bs : kBatchSizes) check_roundtrip(allnull, bs);
}

TEST_CASE("WP-14 round-trip: seeded mixed tables across batch boundaries") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xC0FFEEu);
    for (int iter = 0; iter < 40; ++iter) {
        const std::size_t n = 1 + (rng() % 5000);  // spans many 2048-batches + tail
        std::uniform_int_distribution<int> npct(0, 40);
        const int null_pct = npct(rng);

        Schema s;
        s.fields.emplace_back("i32", Type::I32);
        s.fields.emplace_back("i64", Type::I64);
        s.fields.emplace_back("ts", Type::TS);
        s.fields.emplace_back("f64", Type::F64);
        s.fields.emplace_back("b", Type::BOOL);

        std::vector<OwnedColumn> cols;
        OwnedColumn ci = OwnedColumn::make(Type::I32, n);
        OwnedColumn cj = OwnedColumn::make(Type::I64, n);
        OwnedColumn ct = OwnedColumn::make(Type::TS, n);
        OwnedColumn cf = OwnedColumn::make(Type::F64, n);
        OwnedColumn cb = OwnedColumn::make(Type::BOOL, n);
        auto* pi = reinterpret_cast<std::int32_t*>(ci.mutable_data());
        auto* pj = reinterpret_cast<std::int64_t*>(cj.mutable_data());
        auto* pt = reinterpret_cast<std::int64_t*>(ct.mutable_data());
        auto* pf = reinterpret_cast<double*>(cf.mutable_data());
        auto* pb = reinterpret_cast<std::uint8_t*>(cb.mutable_data());
        std::int64_t ts = static_cast<std::int64_t>(rng());
        for (std::size_t r = 0; r < n; ++r) {
            pi[r] = static_cast<std::int32_t>(rng());
            pj[r] = static_cast<std::int64_t>(rng());
            ts += static_cast<std::int64_t>(rng() % 1000);  // ascending-ish ticks
            pt[r] = ts;
            std::uint64_t fb = rng();  // arbitrary double bit patterns
            std::memcpy(pf + r, &fb, 8);
            pb[r] = rng() & 1u;
            if (null_pct && static_cast<int>(rng() % 100) < null_pct) {
                ci.set_null(r); cj.set_null(r); ct.set_null(r);
                cf.set_null(r); cb.set_null(r);
            }
        }
        cols.push_back(std::move(ci));
        cols.push_back(std::move(cj));
        cols.push_back(std::move(ct));
        cols.push_back(std::move(cf));
        cols.push_back(std::move(cb));
        const Table tbl(s, std::move(cols));
        check_roundtrip(tbl, 64);
        check_roundtrip(tbl, 2048);
        check_roundtrip(tbl, 1 + (rng() % 300));  // odd boundary
    }
}
