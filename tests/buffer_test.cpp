//  WP-1: Buffer ownership/alignment + OwnedColumn validity-precedence tests.

#include <cstddef>
#include <cstdint>

#include "doctest/doctest.h"

#include "core/buffer.h"
#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "simd/aligned_alloc.h"
#include "simd/kernel_convention.h"

using namespace qe;

TEST_CASE("byte_width matches the frozen layout") {
    CHECK(byte_width(Type::I32) == 4);
    CHECK(byte_width(Type::I64) == 8);
    CHECK(byte_width(Type::F64) == 8);
    CHECK(byte_width(Type::BOOL) == 1);
    CHECK(byte_width(Type::TS) == 8);
}

TEST_CASE("Buffer allocates aligned, sized storage; empty buffer is null") {
    Buffer empty;
    CHECK(empty.data() == nullptr);
    CHECK(empty.size() == 0);
    CHECK(empty.empty());
    CHECK(empty.alignment() == 0);

    const std::size_t want_align = simd::required_alignment_bytes();
    CHECK(want_align >= 1);

    Buffer b(1000);
    REQUIRE(b.data() != nullptr);
    CHECK(b.size() == 1000);
    CHECK(b.alignment() == want_align);
    // The runtime-derived alignment contract must actually hold.
    CHECK(reinterpret_cast<std::uintptr_t>(b.data()) % want_align == 0);

    // Writable across its whole extent (ASan/UBSan would flag an overrun).
    std::byte* p = b.data();
    for (std::size_t i = 0; i < b.size(); ++i) p[i] = std::byte{0xAB};
    CHECK(static_cast<unsigned>(p[999]) == 0xAB);
}

TEST_CASE("Buffer is move-only and transfers ownership") {
    Buffer a(64);
    std::byte* raw = a.data();
    Buffer moved(std::move(a));
    CHECK(moved.data() == raw);
    CHECK(moved.size() == 64);
    CHECK(a.data() == nullptr);  // NOLINT(bugprone-use-after-move) — checking moved-from
    CHECK(a.size() == 0);

    Buffer c(8);
    c = std::move(moved);
    CHECK(c.data() == raw);
    CHECK(c.size() == 64);
    CHECK(moved.data() == nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("required_alignment_bytes is a power of two >= runtime vector width") {
    const std::size_t a = simd::required_alignment_bytes();
    CHECK(a >= simd::runtime_vector_bytes());
    CHECK((a & (a - 1)) == 0);  // power of two
}

TEST_CASE("OwnedColumn: no-null column reports all_valid and a null bitmap") {
    OwnedColumn c = OwnedColumn::make(Type::I32, 16);
    auto* d = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (int i = 0; i < 16; ++i) d[i] = i * 7;

    CHECK(c.all_valid());
    Column v = c.view();
    CHECK(v.type == Type::I32);
    CHECK(v.len == 16);
    CHECK(v.all_valid == true);
    // Precedence: all_valid => validity exposed as nullptr (fast path).
    CHECK(v.validity == nullptr);
    CHECK(reinterpret_cast<const std::int32_t*>(v.data)[15] == 15 * 7);
}

TEST_CASE("OwnedColumn: set_null clears all_valid and exposes the bitmap") {
    OwnedColumn c = OwnedColumn::make(Type::F64, 10);
    CHECK(c.all_valid());
    CHECK(c.view().validity == nullptr);

    c.set_null(3);
    c.set_null(7);
    CHECK_FALSE(c.all_valid());

    Column v = c.view();
    CHECK(v.all_valid == false);
    REQUIRE(v.validity != nullptr);  // precedence: nulls present => bitmap shown
    // bit 3 and 7 are null, the rest valid.
    for (std::size_t i = 0; i < 10; ++i) {
        const bool valid = (v.validity[i >> 6] >> (i & 63u)) & 1ull;
        CHECK(valid == (i != 3 && i != 7));
    }
}

TEST_CASE("OwnedColumn::refresh_all_valid recomputes from the bitmap") {
    OwnedColumn c = OwnedColumn::make(Type::I64, 5);
    c.ensure_validity();           // bitmap allocated, all valid
    c.refresh_all_valid();
    CHECK(c.all_valid());          // an all-ones bitmap is all-valid
    CHECK(c.view().validity == nullptr);  // still fast-path

    // Manually clear a bit, then refresh.
    c.set_null(2);
    CHECK_FALSE(c.all_valid());
}
