//  WP-3 (seed of WP-9): the LOGICAL QUERY — one description of a
//  scan -> [filter] -> project query that BOTH the engine (as an operator tree)
//  and the oracle (as SQL for DuckDB / as a direct computation for the reference)
//  execute over the SAME Table. Keeping a single logical description is what
//  makes the differential honest: there is one source of truth for the query,
//  rendered into each backend.
//
//  WP-3 scope: an optional BOOL filter predicate and a non-empty projection list.
//  WP-5 adds an optional GROUP BY (key columns + aggregate specs). Still no
//  JOIN / ORDER BY (those arrive with their WPs); the absence of ORDER BY is why
//  the comparator canonicalizes (D12) — GROUP BY has no inherent row order.
//
//  ONE logical description, rendered into each backend (engine tree / DuckDB SQL
//  / independent reference): that single source of truth is what makes the diff
//  honest. A query is EITHER a projection query (scan->[filter]->project) OR a
//  group-by query (scan->[filter]->aggregate); group_by, when present, wins and
//  `projections` is ignored.
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "expr/expr.h"
#include "ops/aggregate.h"
#include "ops/operator.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/table.h"

namespace qe::oracle {

// A GROUP BY: key columns (child-output indices, in key order; empty => a single
// global aggregate) plus the aggregates to compute (>=1).
struct GroupBy {
    std::vector<std::uint32_t> keys;
    std::vector<AggSpec> aggs;
};

struct LogicalQuery {
    // A BOOL predicate, or a default-constructed (empty) Expr meaning "no WHERE".
    expr::Expr filter;
    // SELECT list (>=1) — used only when there is no GROUP BY.
    std::vector<Projection> projections;
    // Optional GROUP BY; when set, the query is an aggregation, not a projection.
    std::optional<GroupBy> group_by;

    bool has_filter() const { return static_cast<bool>(filter); }
    bool has_group_by() const { return group_by.has_value(); }
};

// The OUTPUT schema of `q` over an input of schema `input` — the single source of
// truth for result column order/types that every backend renders. For a group-by
// query this is the key columns (child name+type, in key order) followed by the
// aggregate columns (out_name + agg_result_type); otherwise the projection list.
Schema query_output_schema(const Schema& input, const LogicalQuery& q);

// Assemble the engine operator tree for `q`: scan(table) -> [filter] ->
// (aggregate | project). The returned operator OWNS its child chain; it borrows
// `table` (which must outlive the tree). batch_size is forwarded to Scan.
std::unique_ptr<Operator> build_engine_pipeline(
    const Table& table, const LogicalQuery& q,
    std::size_t batch_size = Scan::kDefaultBatchSize);

}  // namespace qe::oracle
