//  WP-3 (seed of WP-9): the DuckDB differential backend — the AUTHORITATIVE
//  golden model (decision D16). It loads a Table into an in-memory DuckDB,
//  renders the LogicalQuery to SQL (oracle/sql_render.h), runs it, and reads the
//  result back into a ResultSet so it feeds the IDENTICAL comparator the
//  reference backend does (oracle/result_set.h, D11/D12).
//
//  BUILD GATING: DuckDB is linked ONLY in this oracle test target and ONLY when
//  the amalgamation is staged at third_party/duckdb/ (see that dir's
//  VENDORING.md). CMake defines QE_WITH_DUCKDB and compiles duckdb_oracle.cpp
//  only then; the engine never links DuckDB (from-scratch gate). When not staged,
//  duckdb_available() is false and the differential tests run against the
//  independent reference oracle alone.
#pragma once

#include <stdexcept>

#include "oracle/logical_query.h"
#include "oracle/result_set.h"
#include "ops/table.h"
#include "plan/plan.h"

namespace qe::oracle {

// True iff this build linked DuckDB (QE_WITH_DUCKDB).
constexpr bool duckdb_available() {
#ifdef QE_WITH_DUCKDB
    return true;
#else
    return false;
#endif
}

// Thrown when DuckDB raises on a statement. Per the divergence policy the
// generators never produce a raising query, so this is a defensive backstop: the
// harness treats it as "regenerate this case", never as a silent diff.
struct DuckDBError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Run `q` over `table` through DuckDB and return its result set. Only available
// when duckdb_available(); throws std::logic_error otherwise.
ResultSet run_duckdb(const Table& table, const LogicalQuery& q);

// WP-6: run a two-input equi-join through DuckDB. Loads `probe` and `build` as
// SQL tables, renders the join (oracle/sql_render.h join_sql), runs it, and reads
// the result back into a ResultSet feeding the IDENTICAL comparator. Authoritative
// golden model (D16). Only available when duckdb_available().
ResultSet run_join_duckdb(const Table& probe, const Table& build,
                          const JoinQuery& jq);

// WP-8: run an arbitrary plan::Plan through DuckDB and return its result set.
// Renders the plan to one composable SQL query (oracle/plan_sql.h), loads each
// base Table the plan scans, runs it, and reads back into a ResultSet feeding the
// IDENTICAL comparator. Authoritative golden model (D16). Only available when
// duckdb_available().
ResultSet run_plan_duckdb(const qe::plan::Plan& p);

}  // namespace qe::oracle
