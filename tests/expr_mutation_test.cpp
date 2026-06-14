//  WP-2 GATE: MUTATION SELF-TEST. "A checker that cannot fail proves nothing."
//
//  Three planted bugs, one per §12 expression hotspot (expr/expr_kernels_mutants):
//    1. SIMD tail dropped in a comparison kernel,
//    2. three-valued logic degraded to plain null-if-any,
//    3. the all-valid fast path dropping a real null.
//  For each we DEMONSTRATE the exact check the real suite uses (scalar==vector,
//  the AND truth table, null propagation) catches the mutant, then that the real
//  path passes that same check.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/eval_internal.h"
#include "expr/expr_kernels_mutants.h"
#include "expr/kernels.h"
#include "tests/expr_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::expr;
using qe::expr::test::valid_at;

TEST_CASE("MUTATION 1: comparison SIMD tail bug is caught by scalar==vector") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x7A1Lu);
    const std::size_t n = 65;  // not a whole number of vectors => has a tail
    std::vector<std::int32_t> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::int32_t>(rng());
        b[i] = static_cast<std::int32_t>(rng());
    }
    std::vector<std::uint8_t> ref(n), real(n), mut(n, 0xFF);
    cmp_scalar(CmpOp::Lt, Type::I32, a.data(), b.data(), ref.data(), n);
    cmp_vec(CmpOp::Lt, Type::I32, a.data(), b.data(), real.data(), n);
    mutant::cmp_lt_i32_tailbug_vec(a.data(), b.data(), mut.data(), n);

    CHECK(real == ref);   // the REAL kernel passes the differential
    CHECK(mut != ref);    // the MUTANT is caught: its dropped tail diverges
    // Concretely, the last (tail) lane was never written by the mutant.
    CHECK(mut[n - 1] == 0xFF);
    CHECK(ref[n - 1] != 0xFF);
}

TEST_CASE("MUTATION 2: three-valued AND degraded to null-if-any is caught") {
    // All 9 tri-state combos (0=F,1=N,2=T).
    std::vector<std::uint8_t> a = {0, 0, 0, 2, 2, 2, 1, 1, 1};
    std::vector<std::uint8_t> b = {0, 2, 1, 0, 2, 1, 0, 2, 1};
    const std::size_t n = a.size();
    std::vector<std::uint8_t> ref(n), real(n), mut(n);
    logic_and_scalar(a.data(), b.data(), ref.data(), n);
    logic_and_vec(a.data(), b.data(), real.data(), n);
    mutant::logic_and_twovalued(a.data(), b.data(), mut.data(), n);

    CHECK(real == ref);  // real kernel implements Kleene AND
    CHECK(mut != ref);   // mutant is caught by the truth table

    // The decisive row: FALSE ∧ NULL must be FALSE(0), but the mutant says
    // NULL(1) — exactly the null-propagation hotspot.
    // a=F(0), b=N(1) at index 2:
    CHECK(ref[2] == 0);
    CHECK(mut[2] == 1);
}

TEST_CASE("MUTATION 3: all-valid fast path dropping a real null is caught") {
    // a is all-valid; b carries a null at row 2. The correct combine keeps the
    // null; the mutant short-circuits on a.all_valid and drops it.
    OwnedColumn a = OwnedColumn::make(Type::I32, 4);
    OwnedColumn b = OwnedColumn::make(Type::I32, 4);
    auto* da = reinterpret_cast<std::int32_t*>(a.mutable_data());
    auto* db = reinterpret_cast<std::int32_t*>(b.mutable_data());
    for (int i = 0; i < 4; ++i) { da[i] = i; db[i] = i; }
    b.set_null(2);

    OwnedColumn out_real = OwnedColumn::make(Type::I32, 4);
    OwnedColumn out_mut = OwnedColumn::make(Type::I32, 4);
    propagate_nulls_and(out_real, a.view(), b.view());
    mutant::propagate_nulls_and_allvalidbug(out_mut, a.view(), b.view());

    // Real keeps b's null; mutant wrongly reports the row valid.
    CHECK_FALSE(valid_at(out_real.view(), 2));
    CHECK(valid_at(out_mut.view(), 2));
    CHECK(valid_at(out_real.view(), 2) != valid_at(out_mut.view(), 2));
    // And the real combine left the all-valid rows valid (no false positives).
    CHECK(valid_at(out_real.view(), 0));
    CHECK(valid_at(out_real.view(), 3));
}
