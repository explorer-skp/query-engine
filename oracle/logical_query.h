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
#include "ops/join.h"
#include "ops/operator.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/sort.h"
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
    // Optional ORDER BY (WP-7): sort keys over the query's OUTPUT columns
    // (SortKey.col is the output-column index). When present the result row order
    // is DEFINED, so the differential compares POSITIONALLY (ORDERED mode — see
    // oracle/result_set.h). A Sort is appended atop the projection/aggregate.
    std::optional<std::vector<SortKey>> order_by;

    bool has_filter() const { return static_cast<bool>(filter); }
    bool has_group_by() const { return group_by.has_value(); }
    bool has_order_by() const { return order_by.has_value(); }
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

// ---- WP-6: two-input equi-join ---------------------------------------------
// ONE logical description of a join that BOTH the engine tree and the oracle
// (DuckDB SQL / the independent reference) render from, so the differential is
// honest. The query references two external Tables, the PROBE (left) table and
// the BUILD (right) table; `probe_keys`/`build_keys` are the equi-key column
// indices into each table's schema (in key order, equal length), and `type` is
// INNER or LEFT-OUTER. Output column order is the engine's: ALL probe columns
// then ALL build columns (see ops/join.h).
struct JoinQuery {
    std::vector<std::uint32_t> probe_keys;
    std::vector<std::uint32_t> build_keys;
    JoinType type = JoinType::Inner;
};

// Output schema of a join over the given table schemas: probe fields followed by
// build fields (single source of truth for column order/types every backend
// renders). Names are the children's names verbatim and MAY collide across sides
// — the comparator and renderer use position.
Schema join_output_schema(const Schema& probe, const Schema& build);

// Assemble the engine tree for `jq`: scan(probe) + scan(build) -> HashJoin. The
// returned operator OWNS both scans; it borrows both Tables (which must outlive
// the tree). batch_size is forwarded to BOTH scans.
std::unique_ptr<Operator> build_join_pipeline(
    const Table& probe, const Table& build, const JoinQuery& jq,
    std::size_t batch_size = Scan::kDefaultBatchSize);

}  // namespace qe::oracle
