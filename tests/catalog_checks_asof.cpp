//  WP-12 catalog check — operator orchestration: backward as-of join. This TU
//  brings qe::mutant::AsofMutation + qe::mutant::AsofJoin (tsx/asof_mutants.h) — a
//  DISTINCT-named enum from the join/sort/aggregate Mutation enums, so it cannot
//  clash with them, but per the conflict-free-TU convention it still lives alone.
//
//  The decisive comparison reproduces the §5 as-of boundary hazard: a probe whose
//  timestamp EQUALS a build timestamp must match (the `>=` boundary). The
//  kBoundaryStrict mutant uses `<`, dropping equal-timestamp matches — flagged by
//  the differential (independent reference + DuckDB ASOF JOIN) while the real
//  operator passes.

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "tsx/asof.h"
#include "tsx/asof_mutants.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;
using namespace qe::plan;

Verdict asof_boundary_strict() {
    // build: key 1 @ t={10,20}, key 2 @ t={15}. probe: (1,20) and (2,15) are EXACT
    // boundary ties (must match t=20 / t=15); (1,25) -> t=20. kBoundaryStrict (`<`)
    // drops the equal-timestamp matches => value/row-count divergence.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);  // timestamp (I64 ns-like)
    bs.fields.emplace_back("c2", Type::I64);  // payload
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2}));
    bcols.push_back(i64_col({10, 20, 15}));
    bcols.push_back(i64_col({100, 200, 150}));
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    ps.fields.emplace_back("c1", Type::I64);  // timestamp
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 1, 2}));
    pcols.push_back(i64_col({20, 25, 15}));
    const Table probe(ps, std::move(pcols));

    const std::vector<std::uint32_t> lk{0}, rk{0};
    const std::uint32_t lt = 1, rt = 1;
    const ResultSet ref =
        run_asof_reference(probe, build, lk, rk, lt, rt, tsx::AsofType::Inner,
                           std::nullopt);
    const Plan plan = scan(probe)
                          .asof_join(scan(build), {0}, {0}, 1, 1,
                                     tsx::AsofType::Inner, std::nullopt)
                          .plan();
    const auto duckdb = [&]() { return run_plan_duckdb(plan); };

    // Clean engine = the real operator.
    auto p1 = std::make_unique<Scan>(probe, 2048);
    auto b1 = std::make_unique<Scan>(build, 2048);
    tsx::AsofJoin real(std::move(p1), std::move(b1), lk, rk, lt, rt,
                       tsx::AsofType::Inner, std::nullopt);
    const ResultSet eng = drain_operator(real);

    // Mutant = strict-boundary as-of.
    auto p2 = std::make_unique<Scan>(probe, 2048);
    auto b2 = std::make_unique<Scan>(build, 2048);
    mutant::AsofJoin mut(std::move(p2), std::move(b2), lk, rk, lt, rt,
                         tsx::AsofType::Inner, std::nullopt,
                         mutant::AsofMutation::kBoundaryStrict);
    const ResultSet meng = drain_operator(mut);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
