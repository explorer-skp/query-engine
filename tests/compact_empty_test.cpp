//  WP-9 REGRESSION for carry-forward (1): core/owned_batch.cpp::compact_column
//  ABORTED on n==0 (WP-15 pinned-bug candidate). The out-of-place assert
//  `out.data() != in.data` assumed both buffers are non-null; at n==0 the result
//  Buffer is 0 bytes (data()==nullptr) and an EMPTY input column's data is also
//  nullptr, so `nullptr != nullptr` is false and the assert fired (SIGABRT).
//
//  This is the BEFORE/AFTER regression: with the bug present these cases ABORT the
//  process (ctest reports the crash as a failure); with the fix in place
//  compact_column returns a valid, dense, all-valid EMPTY OwnedColumn of the right
//  type and every CHECK below passes. Confirmed by
//  reverting the one-line `if (n == 0) return out;` guard: the FIRST case aborts.
//
//  The fix is signature-preserving — only the body of compact_column changed; the
//  declaration in core/owned_batch.h is byte-for-byte unchanged (the sole, narrow,
//  reviewer-authorized exception to the frozen-core rule). The matching 0-row
//  PLAN cases (a filter that selects nothing; a global aggregate over empty input),
//  diffed against DuckDB, live in tests/oracle_zero_row_test.cpp.
//
//  No randomness -> plain doctest main (tests/test_main.cpp); nothing to replay.

#include <cstdint>

#include "doctest/doctest.h"

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"

using namespace qe;

namespace {

// An OwnedColumn of `type`/`len` with deterministic payload (so the non-empty
// inputs below are real columns, not just headers). Width-agnostic: writes raw
// bytes through the typed view the caller will not inspect (we only ever compact
// ZERO rows out of it, so the values are irrelevant — only that data() != nullptr).
OwnedColumn filled(Type type, std::size_t len) {
    OwnedColumn c = OwnedColumn::make(type, len);
    auto* p = reinterpret_cast<std::uint8_t*>(c.mutable_data());
    for (std::size_t i = 0; i < len * byte_width(type); ++i)
        p[i] = static_cast<std::uint8_t>(i + 1);
    return c;
}

void check_empty_result(const OwnedColumn& out, Type expect) {
    CHECK(out.type() == expect);
    CHECK(out.len() == 0);
    CHECK(out.all_valid());            // an empty column has no nulls
    const Column v = out.view();
    CHECK(v.len == 0);
    CHECK(v.all_valid);
    CHECK(v.validity == nullptr);      // validity precedence: no nulls => nullptr
}

}  // namespace

// THE crash case before the fix: compacting an EMPTY input column with n==0. Here
// in.data == nullptr and out.data() == nullptr, which tripped the out-of-place
// assert. Must now return a valid empty column for every lane width.
TEST_CASE("compact_column(empty input, n==0) returns a valid empty column") {
    for (const Type t : {Type::BOOL, Type::I32, Type::I64, Type::F64, Type::TS}) {
        OwnedColumn empty = OwnedColumn::make(t, 0);
        const Column in = empty.view();
        REQUIRE(in.data == nullptr);   // the precondition that armed the abort
        const OwnedColumn out = compact_column(in, /*sel=*/nullptr, /*n=*/0);
        check_empty_result(out, t);
    }
}

// The other 0-row shape: a NON-empty input from which a selection vector picks
// ZERO rows (filter-selects-nothing). in.data != nullptr here, so this path did
// NOT abort before the fix, but it must still produce a valid empty column — we
// pin it so a future "optimization" cannot regress it.
TEST_CASE("compact_column(non-empty input, empty selection, n==0) is empty") {
    for (const Type t : {Type::BOOL, Type::I32, Type::I64, Type::F64, Type::TS}) {
        const OwnedColumn src = filled(t, 8);
        const Column in = src.view();
        REQUIRE(in.data != nullptr);
        const SelectionVector empty_sel{/*idx=*/nullptr, /*len=*/0};
        // Both spellings of "no rows": a null sel and a length-0 sel.
        check_empty_result(compact_column(in, nullptr, 0), t);
        check_empty_result(compact_column(in, &empty_sel, 0), t);
    }
}

// A NULL-bearing non-empty input compacted to zero rows must still yield an
// all-valid empty column (no stray validity buffer carried over).
TEST_CASE("compact_column(nullable input, n==0) drops validity, stays empty") {
    OwnedColumn src = OwnedColumn::make(Type::I64, 4);
    src.set_null(1);                   // force a validity bitmap + all_valid=false
    REQUIRE_FALSE(src.all_valid());
    const Column in = src.view();
    REQUIRE(in.validity != nullptr);
    check_empty_result(compact_column(in, nullptr, 0), Type::I64);
}
