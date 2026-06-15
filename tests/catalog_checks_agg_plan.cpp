//  WP-9 catalog checks — operator orchestration: aggregation and plan lowering.
//  aggregate_mutants.h brings qe::mutant::Mutation + qe::mutant::Aggregate;
//  plan_mutants.h brings qe::plan::LowerMutation (no Mutation enum) — so they share
//  a TU, kept apart from join/hashtable's Mutation enums.

#include "tests/catalog_checks.h"

#include <memory>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/aggregate.h"
#include "ops/aggregate_mutants.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "plan/plan.h"
#include "plan/plan_mutants.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;

Verdict agg_fold_null_in_sum() {
    // Three groups {1,2,3}; group 1's value column is ALL NULL. Folding a NULL as 0
    // in SUM/AVG makes group 1 emit 0 (and a non-NULL "seen") instead of NULL.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I32);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({1, 1, 2, 2, 3, 3}));
    cols.push_back(i32_col({0, 0, 5, 6, 7, 8}, /*nulls=*/{0, 1}));
    const Table t(s, std::move(cols));

    LogicalQuery q;
    q.group_by = GroupBy{{0},
                         {AggSpec::count_star("cs"), AggSpec::count(1, "c1"),
                          AggSpec::sum(1, "s1"), AggSpec::avg(1, "av")}};

    auto scan = std::make_unique<Scan>(t, 64);
    auto mut = std::make_unique<mutant::Aggregate>(
        std::move(scan), q.group_by->keys, q.group_by->aggs,
        mutant::Mutation::kFoldNullInSum);
    return single_table_verdict(t, q, std::move(mut), /*ordered=*/false);
}

Verdict plan_drop_sort() {
    // Unsorted input: dropping the Sort during lowering leaves rows out of order,
    // which the POSITIONAL (root-is-Sort) comparator catches.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({3, 1, 2, 5, 4}));
    const Table t(s, std::move(cols));

    const qe::plan::Plan p =
        qe::plan::scan(t).sort({SortKey{0, SortDir::Asc, NullOrder::Last}}).plan();

    const ResultSet ref = run_plan_reference(p);
    const auto duckdb = [&]() { return run_plan_duckdb(p); };

    auto real = p.lower(64);
    const ResultSet eng = drain_operator(*real);

    auto mut = qe::plan::lower_mutant(p, qe::plan::LowerMutation::kDropSort, 64);
    const ResultSet meng = drain_operator(*mut);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/true).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/true);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
