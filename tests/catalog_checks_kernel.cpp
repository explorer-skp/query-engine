//  WP-9 catalog checks — kernel-level hazards + the three new WP-9 mutants. None of
//  these include a conflicting qe::mutant::Mutation header, so they share one TU.
//
//  Covered §12 hazards / mutants:
//    * SIMD tail/remainder          — simd::mutant::all_valid_vec_tailbug
//    * null propagation (Kleene)    — expr::mutant::logic_and_twovalued
//    * all-valid fast path          — expr::mutant::propagate_nulls_and_allvalidbug
//    * selection-vector aliasing    — catalog::mutant::gather_in_place_aliased
//    * float->int round via +0.5    — catalog::mutant::cast_f64_to_i64_plus_half
//    * aggregation accumulator OF   — catalog::mutant::sum_i64_wrapping_i32

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "core/validity.h"
#include "expr/eval_internal.h"
#include "expr/expr.h"
#include "expr/expr_kernels_mutants.h"
#include "expr/kernels.h"
#include "simd/validity_kernels.h"
#include "simd/validity_kernels_mutants.h"

#include "ops/aggregate.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"

#include "tests/catalog_check_util.h"
#include "tests/catalog_mutants.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;

Verdict simd_tail() {
    // 65 validity bits => a partial final word past bit 63; the null at bit 64
    // lives in that tail word — exactly what a dropped-tail kernel skips.
    const std::size_t nbits = 65;
    std::vector<std::uint64_t> w(validity::words(nbits), 0);
    validity::fill_all_valid(w.data(), nbits);
    validity::set_bit(w.data(), 64, /*valid=*/false);

    const bool real = simd::all_valid_vec(w.data(), nbits);
    const bool mut = simd::mutant::all_valid_vec_tailbug(w.data(), nbits);
    Verdict v;
    v.clean_passes = (real == false);   // real kernel sees the tail null
    v.mutant_flagged = (mut == true);   // mutant misses it (skips the partial word)
    v.detail = "all_valid_vec=" + std::to_string(real) + " tailbug=" +
               std::to_string(mut) + " (truth: NOT all-valid)";
    return v;
}

Verdict null_propagation() {
    // All 9 tri-state combos (0=F,1=N,2=T); the decisive row is FALSE ∧ NULL == F.
    const std::vector<std::uint8_t> a = {0, 0, 0, 2, 2, 2, 1, 1, 1};
    const std::vector<std::uint8_t> b = {0, 2, 1, 0, 2, 1, 0, 2, 1};
    const std::size_t n = a.size();
    std::vector<std::uint8_t> ref(n), real(n), mut(n);
    logic_and_scalar(a.data(), b.data(), ref.data(), n);
    logic_and_vec(a.data(), b.data(), real.data(), n);
    expr::mutant::logic_and_twovalued(a.data(), b.data(), mut.data(), n);
    Verdict v;
    v.clean_passes = (real == ref);
    v.mutant_flagged = (mut != ref);
    v.detail = "FALSE∧NULL ref=" + std::to_string(ref[2]) + " mutant=" +
               std::to_string(mut[2]) + " (must be 0=FALSE)";
    return v;
}

Verdict all_valid_fastpath() {
    // a all-valid; b null at row 2. Correct AND-of-validity keeps the null; the
    // mutant short-circuits on a.all_valid and drops it.
    OwnedColumn a = OwnedColumn::make(Type::I32, 4);
    OwnedColumn b = OwnedColumn::make(Type::I32, 4);
    auto* da = reinterpret_cast<std::int32_t*>(a.mutable_data());
    auto* db = reinterpret_cast<std::int32_t*>(b.mutable_data());
    for (int i = 0; i < 4; ++i) { da[i] = i; db[i] = i; }
    b.set_null(2);

    OwnedColumn out_real = OwnedColumn::make(Type::I32, 4);
    OwnedColumn out_mut = OwnedColumn::make(Type::I32, 4);
    propagate_nulls_and(out_real, a.view(), b.view());
    expr::mutant::propagate_nulls_and_allvalidbug(out_mut, a.view(), b.view());

    const auto valid = [](const Column& c, std::size_t i) {
        return c.all_valid || c.validity == nullptr ||
               validity::get_bit(c.validity, i);
    };
    Verdict v;
    v.clean_passes = !valid(out_real.view(), 2);  // real keeps b's null
    v.mutant_flagged = valid(out_mut.view(), 2);  // mutant wrongly reports valid
    v.detail = "row2 valid: real=" + std::to_string(valid(out_real.view(), 2)) +
               " mutant=" + std::to_string(valid(out_mut.view(), 2));
    return v;
}

Verdict selection_aliasing() {
    // A reversing permutation: out-of-place compact is correct {40,30,20,10}; the
    // in-place aliased gather overwrites a slot before a later index reads it.
    const std::vector<std::int64_t> data = {10, 20, 30, 40};
    const std::vector<std::uint32_t> perm = {3, 2, 1, 0};
    const std::vector<std::int64_t> expect = {40, 30, 20, 10};

    OwnedColumn src = i64_col(data);
    const SelectionVector sel{perm.data(), perm.size()};
    const OwnedColumn out = compact_column(src.view(), &sel, perm.size());  // real
    const auto* od = reinterpret_cast<const std::int64_t*>(out.view().data);
    bool real_ok = (out.len() == data.size());
    for (std::size_t i = 0; real_ok && i < data.size(); ++i)
        real_ok = (od[i] == expect[i]);

    std::vector<std::int64_t> buf = data;  // mutant works in-place
    catalog::mutant::gather_in_place_aliased(buf.data(), perm.data(), perm.size());
    const bool mut_ok = (buf == expect);

    Verdict v;
    v.clean_passes = real_ok;      // out-of-place compact matches the reference
    v.mutant_flagged = !mut_ok;    // in-place aliasing corrupts the result
    v.detail = "in-place gather over a reversed permutation corrupted the result";
    return v;
}

Verdict float_to_int_round() {
    // Integral doubles with |x| >= 2^52 (odd, so +0.5 rounds to the WRONG integer).
    constexpr double k2_52 = 4503599627370496.0;  // 2^52
    const std::vector<double> vals = {k2_52 + 1.0, k2_52 + 3.0, -(k2_52 + 1.0)};

    Schema s;
    s.fields.emplace_back("c0", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(f64_col(vals));
    const Table t(s, std::move(cols));

    LogicalQuery q;
    q.projections.push_back({"p", cast(col(Type::F64, 0), Type::I64)});

    const ResultSet ref = run_reference(t, q);  // correct round-half-away cast
    const auto duckdb = [&]() { return run_duckdb(t, q); };

    auto real = build_engine_pipeline(t, q);
    const ResultSet eng = drain_operator(*real);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;

    // The mutant result: the buggy +0.5 cast applied to the same column.
    std::vector<std::int64_t> buggy(vals.size());
    catalog::mutant::cast_f64_to_i64_plus_half(vals.data(), buggy.data(),
                                               vals.size());
    ResultSet mut;
    mut.types = {Type::I64};
    for (std::int64_t x : buggy) mut.rows.push_back({Cell{false, x, 0.0}});

    const DiffResult md = diff_with_duckdb(mut, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

Verdict aggregation_overflow() {
    // Eight values of 3e8 => SUM = 2.4e9 (> 2^31, fits I64 and DuckDB BIGINT). The
    // mutant's i32 accumulator wraps to a negative number.
    const std::vector<std::int64_t> vals(8, 300'000'000LL);
    Schema s;
    s.fields.emplace_back("c0", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i64_col(vals));
    const Table t(s, std::move(cols));

    LogicalQuery q;
    q.group_by = GroupBy{{}, {AggSpec::sum(0, "s")}};  // global SUM

    const ResultSet ref = run_reference(t, q);
    const auto duckdb = [&]() { return run_duckdb(t, q); };

    auto real = build_engine_pipeline(t, q);
    const ResultSet eng = drain_operator(*real);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;

    const std::int64_t buggy =
        catalog::mutant::sum_i64_wrapping_i32(vals.data(), vals.size());
    ResultSet mut;
    mut.types = {agg_result_type(AggFunc::Sum, Type::I64)};  // I64
    mut.rows.push_back({Cell{false, buggy, 0.0}});

    const DiffResult md = diff_with_duckdb(mut, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = "i32-accumulator SUM=" + std::to_string(buggy) +
               " (true SUM=2400000000)";
    return v;
}

}  // namespace qe::catalog::checks
