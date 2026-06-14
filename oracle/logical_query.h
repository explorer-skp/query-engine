//  WP-3 (seed of WP-9): the LOGICAL QUERY — one description of a
//  scan -> [filter] -> project query that BOTH the engine (as an operator tree)
//  and the oracle (as SQL for DuckDB / as a direct computation for the reference)
//  execute over the SAME Table. Keeping a single logical description is what
//  makes the differential honest: there is one source of truth for the query,
//  rendered into each backend.
//
//  WP-3 scope: an optional BOOL filter predicate and a non-empty projection list.
//  No GROUP BY / JOIN / ORDER BY yet (those arrive with their WPs); the absence
//  of ORDER BY is why the comparator canonicalizes (D12).
#pragma once

#include <memory>
#include <vector>

#include "expr/expr.h"
#include "ops/operator.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/table.h"

namespace qe::oracle {

struct LogicalQuery {
    // A BOOL predicate, or a default-constructed (empty) Expr meaning "no WHERE".
    expr::Expr filter;
    // SELECT list (>=1). Project requires named columns.
    std::vector<Projection> projections;

    bool has_filter() const { return static_cast<bool>(filter); }
};

// Assemble the engine operator tree scan(table) -> [filter] -> project for `q`.
// The returned operator OWNS its child chain; it borrows `table` (which must
// outlive the tree). batch_size is forwarded to Scan (default = D4 2048).
std::unique_ptr<Operator> build_engine_pipeline(
    const Table& table, const LogicalQuery& q,
    std::size_t batch_size = Scan::kDefaultBatchSize);

}  // namespace qe::oracle
