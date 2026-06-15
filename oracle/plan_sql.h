//  WP-8: render a plan::Plan to the SQL the DuckDB oracle runs — the second half
//  of WP-8's single-source-of-truth (the SAME Plan lowers to the engine tree and
//  renders to this SQL, so engine and DuckDB provably execute one query). It has
//  NO DuckDB dependency (pure string building, unit-testable on its own); the
//  DuckDB runner that consumes it lives in oracle/duckdb_oracle.* (run_plan_duckdb).
//
//  COMPOSABILITY. A plan is an arbitrary tree, so each node is rendered as a
//  SELECT over a SUBQUERY whose output columns are aliased to canonical positional
//  names o0..oN (matching the node's output schema). This sidesteps the two
//  hazards a flat renderer would hit: (1) join column-name COLLISIONS (both sides
//  may have a "c0"), and (2) the expr IR referencing columns by INDEX — every
//  parent resolves an index i to the unambiguous name "oi". It reuses the shared
//  expr->SQL / agg-call / order-by helpers from oracle/sql_render.h (does not
//  duplicate them).
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ops/table.h"
#include "plan/plan.h"

namespace qe::oracle {

struct PlanSql {
    // A single SELECT whose result columns are o0..oN, matching
    // plan.output_schema() positionally (types are applied on read-back).
    std::string sql;
    // The distinct base Tables the query scans, each with the SQL name to
    // CREATE+INSERT it under, in deterministic discovery order.
    std::vector<std::pair<std::string, const qe::Table*>> tables;
};

// Render `p` to one composable SQL query plus the base tables it needs.
PlanSql render_plan_sql(const qe::plan::Plan& p);

}  // namespace qe::oracle
