//  WP-10b MUTATION SELF-TEST (RIGOR.md rule 4). The parallel differential is only
//  credible if it CAN fail: here we plant a parallelism bug
//  (qe::mutant::ParallelMutation::kMergeDropPartial — the cross-worker partial-
//  aggregate merge drops partials) and SHOW that
//    * the REAL parallel driver passes the differential at 1/2/4/8 threads, and
//    * the MUTANT passes at 1 thread (the bug needs >1 worker) but is FLAGGED at
//      2/4/8 threads — by parallel-vs-single-thread AND parallel-vs-reference (and
//      DuckDB when staged).
//
//  The scenario is fixed/deterministic: many rows, few groups, tiny morsels => every
//  group spans several morsels and (at >1 thread) several workers, so the dropped-
//  partial bug under-counts SUM/COUNT and corrupts AVG.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"

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

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using qe::exec::ParallelConfig;
using qe::exec::run_plan_parallel;
using qe::mutant::ParallelMutation;
using qe::mutant::run_plan_parallel_mutant;

namespace {

// A GROUP BY plan over many rows but few keys, so groups span morsels/workers.
struct Fixture {
    Table table;
    Plan plan;
};

Fixture make_fixture() {
    constexpr std::size_t N = 4096;  // several morsels at tiny morsel sizes
    Schema s;
    s.fields.emplace_back("k", Type::I32);   // group key (4 distinct)
    s.fields.emplace_back("v", Type::I64);   // sum/min/max/avg input
    std::vector<OwnedColumn> cols;
    OwnedColumn k = OwnedColumn::make(Type::I32, N);
    OwnedColumn v = OwnedColumn::make(Type::I64, N);
    auto* kd = reinterpret_cast<std::int32_t*>(k.mutable_data());
    auto* vd = reinterpret_cast<std::int64_t*>(v.mutable_data());
    for (std::size_t i = 0; i < N; ++i) {
        kd[i] = static_cast<std::int32_t>(i % 4);    // 4 groups
        vd[i] = static_cast<std::int64_t>(i % 100);  // bounded, overflow-safe
    }
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    Table t(s, std::move(cols));
    // NOTE: build the plan over the table by value AFTER moving — caller keeps the
    // Table stable; we return both so the Scan's borrow stays valid.
    return Fixture{std::move(t), Plan{}};
}

Plan group_plan(const Table& t) {
    return scan(t)
        .aggregate({0}, {AggSpec::count_star("n"), AggSpec::count(1, "cv"),
                         AggSpec::sum(1, "sv"), AggSpec::avg(1, "av"),
                         AggSpec::min(1, "mn"), AggSpec::max(1, "mx")})
        .plan();
}

}  // namespace

TEST_CASE("WP-10b mutation: real driver clean; merge-drop mutant flagged >1 thread") {
    Fixture fx = make_fixture();
    const Table& t = fx.table;
    const Plan p = group_plan(t);

    const ResultSet ref = run_plan_reference(p);
    const bool have_duck = duckdb_available();
    ResultSet duck;
    if (have_duck) duck = run_plan_duckdb(p);

    const std::size_t kMorsel = 64;  // tiny => groups span many morsels

    for (unsigned threads : {1u, 2u, 4u, 8u}) {
        ParallelConfig cfg;
        cfg.threads = threads;
        cfg.morsel_rows = kMorsel;

        // --- the REAL driver is CLEAN at every thread count ---
        const ResultSet real = run_plan_parallel(p, cfg);
        CHECK_MESSAGE(compare_result_sets(real, ref).equal,
                      "real parallel != reference at threads=" << threads);
        if (have_duck)
            CHECK_MESSAGE(compare_result_sets(real, duck).equal,
                          "real parallel != DuckDB at threads=" << threads);

        // --- the MUTANT: clean at 1 thread, FLAGGED at >1 ---
        const ResultSet mut =
            run_plan_parallel_mutant(p, cfg, ParallelMutation::kMergeDropPartial);
        const bool flagged_vs_ref = !compare_result_sets(mut, ref).equal;
        const bool flagged_vs_single =
            !compare_result_sets(mut, real).equal;  // real == single logically

        if (threads == 1) {
            CHECK_MESSAGE(!flagged_vs_ref,
                          "mutant should be CLEAN at 1 thread (bug needs >1 worker)");
        } else {
            CHECK_MESSAGE(flagged_vs_ref,
                          "MUTANT NOT FLAGGED vs reference at threads=" << threads
                          << " (checker cannot fail!)");
            CHECK_MESSAGE(flagged_vs_single,
                          "MUTANT NOT FLAGGED vs single-thread at threads="
                          << threads);
            if (have_duck)
                CHECK_MESSAGE(!compare_result_sets(mut, duck).equal,
                              "MUTANT NOT FLAGGED vs DuckDB at threads=" << threads);
        }
    }
}
