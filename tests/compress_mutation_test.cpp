//  WP-14 MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant deliberately-broken DECODERS
//  (tsx/compress_mutants.{h,cpp}) — each a faithful copy of the real streaming decode
//  over the SAME EncodedColumn format + zig-zag kernel, with exactly one bad step —
//  and SHOW the compressed-scan differential (independent plan reference + DuckDB
//  when staged) CATCHES every mutant, while the REAL decoder PASSES the identical
//  diff.
//
//  Planted decode bit-slips:
//    * kDropSecondDerivative   — delta-of-delta omits the second-difference term, so
//      reconstructed TS/I64 series DRIFT.
//    * kGorillaLeadingZerosOff — Gorilla leading-zero count off by one, so a
//      reconstructed double is wrong by bits.
//
//  Replay:  ./compress_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/operator.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tsx/compress.h"
#include "tsx/compress_mutants.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::tsx;

namespace {

OwnedColumn i64c(Type t, const std::vector<std::int64_t>& v,
                 const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(t, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}
OwnedColumn f64c(const std::vector<double>& v,
                 const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::F64, v.size());
    auto* d = reinterpret_cast<double*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

// A tick-like table: an irregular TS series (so delta-of-delta is non-trivial), an
// I64 payload, and an F64 price series (so Gorilla blocks carry leading zeros).
std::unique_ptr<Table> make_src() {
    Schema s;
    s.fields.emplace_back("ts", Type::TS);
    s.fields.emplace_back("v", Type::I64);
    s.fields.emplace_back("px", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i64c(Type::TS, {10, 20, 35, 55, 80, 110, 145, 145, 200, 260},
                        /*nulls=*/{7}));
    cols.push_back(i64c(Type::I64, {1, -2, 3, -4, 5, -6, 7, -8, 9, -10}));
    cols.push_back(f64c({1.0, 2.0, 1.5, 100.0, 0.25, 99.5, 1.0, 2.5, 1e9, 0.001},
                        /*nulls=*/{4}));
    return std::make_unique<Table>(s, std::move(cols));
}

// Diff a drained engine result against the independent reference (always) and DuckDB
// (when staged) — the SAME escalation every per-WP differential uses.
DiffResult diff_vs_oracles(const ResultSet& eng, const Plan& plan) {
    const ResultSet ref = run_plan_reference(plan);
    DiffResult d = compare_result_sets(eng, ref, /*ordered=*/false);
    if (d.equal && duckdb_available()) {
        try {
            d = compare_result_sets(eng, run_plan_duckdb(plan), false);
        } catch (const DuckDBError&) { /* divergence backstop */ }
    }
    return d;
}

}  // namespace

TEST_CASE("MUTATION: the REAL compressed scan passes the differential") {
    auto src = make_src();
    const CompressedTable ct = CompressedTable::encode(*src);
    const Plan plan = compressed_scan(ct, *src).plan();
    for (std::size_t bs : {3u, 4u, 2048u}) {
        tsx::CompressedScan real(ct, bs);
        const ResultSet eng = drain_operator(real);
        CHECK(diff_vs_oracles(eng, plan).equal);
    }
}

TEST_CASE("MUTATION: kDropSecondDerivative (TS drift) is CAUGHT") {
    auto src = make_src();
    const CompressedTable ct = CompressedTable::encode(*src);
    const Plan plan = compressed_scan(ct, *src).plan();
    mutant::CompressedScan mut(ct, mutant::CompressMutation::kDropSecondDerivative,
                               2048);
    const ResultSet meng = drain_operator(mut);
    const DiffResult d = diff_vs_oracles(meng, plan);
    CHECK_FALSE(d.equal);
    MESSAGE("kDropSecondDerivative caught: " << d.message);
}

TEST_CASE("MUTATION: kGorillaLeadingZerosOff (double wrong by bits) is CAUGHT") {
    auto src = make_src();
    const CompressedTable ct = CompressedTable::encode(*src);
    const Plan plan = compressed_scan(ct, *src).plan();
    mutant::CompressedScan mut(
        ct, mutant::CompressMutation::kGorillaLeadingZerosOff, 2048);
    const ResultSet meng = drain_operator(mut);
    const DiffResult d = diff_vs_oracles(meng, plan);
    CHECK_FALSE(d.equal);
    MESSAGE("kGorillaLeadingZerosOff caught: " << d.message);
}
