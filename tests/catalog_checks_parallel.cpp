//  WP-10b catalog check — operator orchestration: PARALLEL execution. This TU brings
//  qe::mutant::ParallelMutation (exec/parallel_mutants.h) — a DISTINCT-named enum
//  from the join/sort/aggregate/asof/window Mutation enums, so it cannot clash, but
//  per the conflict-free-TU convention it still lives alone.
//
//  The decisive comparison reproduces the §5 parallelism hazard the brief names: a
//  cross-worker partial-aggregate merge that DROPS one partial. Over a GROUP BY with
//  few keys and tiny morsels every group spans several workers at 4 threads, so the
//  kMergeDropPartial mutant under-counts SUM/COUNT and corrupts AVG — flagged by the
//  differential (parallel vs reference + DuckDB) while the real parallel driver
//  passes at the same thread count.

#include "tests/catalog_checks.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "exec/parallel.h"
#include "exec/parallel_mutants.h"
#include "ops/aggregate.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;
using namespace qe::plan;

Verdict parallel_merge_drop_partial() {
    // 1024 rows, 4 groups, value 0..99 (overflow-safe). Tiny morsels (64 rows) =>
    // every group spans many morsels and (at 4 threads) several workers.
    constexpr std::size_t N = 1024;
    Schema s;
    s.fields.emplace_back("k", Type::I32);
    s.fields.emplace_back("v", Type::I64);
    std::vector<OwnedColumn> cols;
    OwnedColumn k = OwnedColumn::make(Type::I32, N);
    OwnedColumn v = OwnedColumn::make(Type::I64, N);
    auto* kd = reinterpret_cast<std::int32_t*>(k.mutable_data());
    auto* vd = reinterpret_cast<std::int64_t*>(v.mutable_data());
    for (std::size_t i = 0; i < N; ++i) {
        kd[i] = static_cast<std::int32_t>(i % 4);
        vd[i] = static_cast<std::int64_t>(i % 100);
    }
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    const Table t(s, std::move(cols));

    const Plan p =
        scan(t)
            .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(1, "sv"),
                             AggSpec::avg(1, "av")})
            .plan();

    const ResultSet ref = run_plan_reference(p);
    const auto duckdb = [&]() { return run_plan_duckdb(p); };

    qe::exec::ParallelConfig cfg;
    cfg.threads = 4;
    cfg.morsel_rows = 64;

    const ResultSet clean = qe::exec::run_plan_parallel(p, cfg);
    const ResultSet mut = qe::mutant::run_plan_parallel_mutant(
        p, cfg, qe::mutant::ParallelMutation::kMergeDropPartial);

    Verdict verdict;
    verdict.clean_passes = diff_with_duckdb(clean, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(mut, ref, duckdb, /*ordered=*/false);
    verdict.mutant_flagged = !md.equal;
    verdict.detail = md.message;
    return verdict;
}

}  // namespace qe::catalog::checks
