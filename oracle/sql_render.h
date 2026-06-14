//  WP-3 (seed of WP-9): render a LogicalQuery to the SQL the DuckDB oracle runs.
//  This is the SINGLE source of truth that turns one logical query into SQL, so
//  the engine and DuckDB provably execute the same query. It has NO DuckDB
//  dependency (pure string building) and is unit-testable on its own; the DuckDB
//  runner that consumes it lives in oracle/duckdb_oracle.* (staged separately).
//
//  Type mapping (engine Type -> DuckDB SQL type):
//    I32 -> INTEGER, I64 -> BIGINT, F64 -> DOUBLE, BOOL -> BOOLEAN,
//    TS  -> BIGINT  (engine TS is int64 ns; we compare the raw ns integer and do
//                    not exercise DuckDB temporal types in WP-3 — exact + simple).
//
//  Each projection is wrapped CAST((expr) AS <its expr.type()>) so the DuckDB
//  result column type EXACTLY matches the engine's, making the read-back
//  unambiguous. The cast is identity/widening (the value already has that type),
//  so it never raises.
#pragma once

#include <string>

#include "core/types.h"
#include "expr/expr.h"
#include "oracle/logical_query.h"

namespace qe::oracle {

// DuckDB SQL type name for an engine Type.
std::string sql_type(Type t);

// Render an expression to SQL, resolving column refs against `schema`'s names.
std::string expr_to_sql(const expr::Expr& e, const Schema& schema);

// "CREATE TABLE <name>(c0 TYPE, c1 TYPE, ...)" for `schema`.
std::string create_table_sql(const std::string& name, const Schema& schema);

// "SELECT CAST((p0expr) AS T0) AS p0, ... FROM <name> [WHERE filter]".
std::string select_sql(const std::string& name, const Schema& schema,
                       const LogicalQuery& q);

// WP-6: render a join to SQL the DuckDB oracle runs:
//   "SELECT CAST(p.<c> AS T) AS o0, ... , CAST(b.<c> AS T) AS oK, ...
//      FROM <probe_name> AS p [LEFT] JOIN <build_name> AS b
//        ON p.<pk0> = b.<bk0> [AND ...]"
// Probe columns first, then build columns, aliased o0..oN by POSITION (so the
// identical c0/c1 names on the two sides never collide). Each output column is
// wrapped CAST(.. AS <its type>) so the DuckDB result column type matches the
// engine's exactly (the WP-5 CAST precedent). NULL keys never match because SQL
// `=` is NULL on a NULL operand and JOIN ON treats that as non-match — exactly
// NullPolicy::kNeverMatch.
std::string join_sql(const Schema& probe, const Schema& build,
                     const JoinQuery& jq, const std::string& probe_name,
                     const std::string& build_name);

}  // namespace qe::oracle
