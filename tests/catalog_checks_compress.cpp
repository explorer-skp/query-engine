//  WP-14 catalog check — operator orchestration: the compressed scan's decode/scan
//  wiring. This TU brings qe::mutant::CompressMutation + qe::mutant::CompressedScan
//  (tsx/compress_mutants.h) — a DISTINCT-named enum from the join/sort/aggregate/asof
//  Mutation enums, so it cannot clash, but per the conflict-free-TU convention it
//  still lives alone.
//
//  The decisive comparison reproduces the §5 codec hazard: a lossless decode must
//  reproduce the source byte-for-byte. The kDropSecondDerivative mutant omits the
//  delta-of-delta second-difference term, so the reconstructed TS series DRIFTS —
//  flagged by the differential (independent plan reference + DuckDB over the
//  decompressed source) while the real decoder passes.

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tsx/compress.h"
#include "tsx/compress_mutants.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;
using namespace qe::plan;

Verdict compress_decode_drift() {
    // An irregular TS series (so delta-of-delta is non-trivial) + an I64 payload.
    // kDropSecondDerivative drifts the reconstructed timestamps => row-value diff.
    Schema s;
    s.fields.emplace_back("ts", Type::TS);
    s.fields.emplace_back("v", Type::I64);
    std::vector<OwnedColumn> cols;
    {
        OwnedColumn t = OwnedColumn::make(Type::TS, 6);
        auto* d = reinterpret_cast<std::int64_t*>(t.mutable_data());
        const std::int64_t v[] = {10, 20, 35, 55, 80, 110};  // increasing gaps
        for (int i = 0; i < 6; ++i) d[i] = v[i];
        cols.push_back(std::move(t));
    }
    cols.push_back(i64_col({100, 200, 300, 400, 500, 600}));
    auto src = std::make_unique<Table>(s, std::move(cols));

    const tsx::CompressedTable ct = tsx::CompressedTable::encode(*src);
    const Plan plan = compressed_scan(ct, *src).plan();
    const ResultSet ref = run_plan_reference(plan);
    const auto duckdb = [&]() { return run_plan_duckdb(plan); };

    // Clean engine = the real decoder.
    tsx::CompressedScan real(ct, 2048);
    const ResultSet eng = drain_operator(real);

    // Mutant = drop-second-derivative decode.
    mutant::CompressedScan mut(
        ct, mutant::CompressMutation::kDropSecondDerivative, 2048);
    const ResultSet meng = drain_operator(mut);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
