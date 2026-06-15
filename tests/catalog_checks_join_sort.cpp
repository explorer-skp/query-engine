//  WP-9 catalog checks — operator orchestration: hash-join and sort. join_mutants.h
//  brings qe::mutant::Mutation + qe::mutant::HashJoin; sort_mutants.h brings the
//  distinct-named qe::mutant::SortMutation + qe::mutant::Sort — no name clash, so
//  the two share a TU (but neither may meet aggregate/hashtable's Mutation enum).

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/join.h"
#include "ops/join_mutants.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/sort_mutants.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;

Verdict join_drop_match() {
    // Single-key INNER join: probe {1,2,3} vs build {1->{10,11}, 2->{20}}; key 3
    // misses. Dropping one matched output row makes the engine short a row.
    Schema bs;
    bs.fields.emplace_back("c0", Type::I32);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(i32_col({1, 1, 2}));
    bcols.push_back(i64_col({10, 11, 20}));
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::I32);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(i32_col({1, 2, 3}));
    const Table probe(ps, std::move(pcols));

    const JoinQuery jq{{0}, {0}, JoinType::Inner};
    const ResultSet ref = run_join_reference(probe, build, jq);
    const auto duckdb = [&]() { return run_join_duckdb(probe, build, jq); };

    auto real = build_join_pipeline(probe, build, jq);
    const ResultSet eng = drain_operator(*real);

    auto ps_op = std::make_unique<Scan>(probe, 2048);
    auto bs_op = std::make_unique<Scan>(build, 2048);
    mutant::HashJoin j(std::move(ps_op), std::move(bs_op), jq.probe_keys,
                       jq.build_keys, jq.type, mutant::Mutation::kDropProbeMatch);
    const ResultSet meng = drain_operator(j);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

Verdict sort_desc_as_asc() {
    // Unique integer keys (incl. negatives) => DESC is a TOTAL order, so the
    // POSITIONAL differential bites a DESC-sorted-as-ASC mutant.
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col({5, -3, 11, 7, 2, -9, 4, 100}));
    cols.push_back(i64_col({1, 2, 3, 4, 5, 6, 7, 8}));
    const Table t(s, std::move(cols));

    // Project every column as-is, then ORDER BY c0 DESC (matches the oracle output
    // schema for a Scan->Sort tree).
    LogicalQuery q;
    for (std::uint32_t i = 0; i < s.fields.size(); ++i)
        q.projections.push_back(
            {"p" + std::to_string(i), col(s.fields[i].second, i)});
    q.order_by = std::vector<SortKey>{SortKey{0, SortDir::Desc, NullOrder::Last}};

    auto scan = std::make_unique<Scan>(t, 64);
    auto mut = std::make_unique<mutant::Sort>(std::move(scan), *q.order_by,
                                              mutant::SortMutation::kDescSortsAsc);
    return single_table_verdict(t, q, std::move(mut), /*ordered=*/true);
}

}  // namespace qe::catalog::checks
