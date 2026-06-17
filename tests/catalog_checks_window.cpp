//  WP-13 catalog check — operator orchestration: windowed / time-bucketed
//  aggregation. This TU brings qe::mutant::WindowMutation + qe::mutant::Window
//  (tsx/window_mutants.h) — a DISTINCT-named enum from the join/sort/aggregate/asof
//  Mutation enums, so it cannot clash with them, but per the conflict-free-TU
//  convention it still lives alone.
//
//  The decisive comparison reproduces the §5 window-frame boundary hazard: a sliding
//  ROWS BETWEEN P PRECEDING AND CURRENT ROW frame must reach EXACTLY P preceding
//  rows. The kFrameOffByOne mutant reaches P+1, so every running aggregate over a
//  partition with more than P+1 rows diverges — flagged by the differential
//  (independent reference + DuckDB window function) while the real operator passes.

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tsx/window.h"
#include "tsx/window_mutants.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;
using namespace qe::plan;

Verdict window_frame_off_by_one() {
    // One partition (key 1) of FOUR time-ascending rows, P=1. The correct running
    // SUM at the last row covers the 2 most-recent rows; kFrameOffByOne (P+1) covers
    // 3 => a value/row-content divergence the differential catches.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);  // partition key
    s.fields.emplace_back("c1", Type::I64);  // timestamp (I64 ns-like, >= 0, distinct)
    s.fields.emplace_back("c2", Type::I64);  // payload
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 1, 1}));
    cols.push_back(i64_col({0, 10, 20, 30}));
    cols.push_back(i64_col({5, 6, 7, 8}));
    const Table input(s, std::move(cols));

    const std::vector<std::uint32_t> keys{0};
    const std::uint32_t time = 1;
    const std::int64_t P = 1;
    const std::vector<AggSpec> aggs{AggSpec::count_star("n"),
                                    AggSpec::sum(2, "s"), AggSpec::min(2, "mn"),
                                    AggSpec::max(2, "mx")};

    const ResultSet ref =
        run_window_reference(input, tsx::WindowMode::Sliding, keys, time, P, aggs);
    const Plan plan = scan(input)
                          .window_sliding({0}, 1, P, aggs)
                          .plan();
    const auto duckdb = [&]() { return run_plan_duckdb(plan); };

    // Clean engine = the real operator.
    auto sc1 = std::make_unique<Scan>(input, 2048);
    tsx::Window real(std::move(sc1), tsx::WindowMode::Sliding, keys, time, P, aggs);
    const ResultSet eng = drain_operator(real);

    // Mutant = frame reaches P+1 preceding rows.
    auto sc2 = std::make_unique<Scan>(input, 2048);
    mutant::Window mut(std::move(sc2), tsx::WindowMode::Sliding, keys, time, P, aggs,
                       mutant::WindowMutation::kFrameOffByOne);
    const ResultSet meng = drain_operator(mut);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
