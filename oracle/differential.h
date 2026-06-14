//  WP-3 (seed of WP-9): the DIFFERENTIAL RUNNER. Given a Table and a
//  LogicalQuery it (1) runs the query through the ENGINE operator tree
//  scan->[filter]->project, draining it to a ResultSet, (2) runs the SAME query
//  through an ORACLE to a ResultSet, and (3) compares them with the frozen
//  D11/D12 contract (oracle/result_set.h).
//
//  The oracle is a pluggable function `(Table, LogicalQuery) -> ResultSet`, so
//  the identical runner + comparator serve every backend:
//    * run_vs_reference()  — the always-available independent reference oracle
//      (oracle/reference_oracle.h). This is what CI runs today.
//    * run_vs_duckdb()     — DuckDB SQL (oracle/sql_oracle.h), compiled only when
//      the DuckDB amalgamation is staged (QE_WITH_DUCKDB). The AUTHORITATIVE
//      golden model per decision D16; it plugs into this same runner.
#pragma once

#include <functional>

#include "oracle/logical_query.h"
#include "oracle/result_set.h"
#include "ops/scan.h"
#include "ops/table.h"

namespace qe::oracle {

using OracleFn = std::function<ResultSet(const Table&, const LogicalQuery&)>;

// Run engine-vs-oracle and return the verdict. batch_size is forwarded to Scan
// (varying it across runs exercises batch-boundary / tail handling).
DiffResult run_differential(const Table& table, const LogicalQuery& q,
                            const OracleFn& oracle,
                            std::size_t batch_size = Scan::kDefaultBatchSize);

// Convenience: engine vs the independent reference oracle.
DiffResult run_vs_reference(const Table& table, const LogicalQuery& q,
                            std::size_t batch_size = Scan::kDefaultBatchSize);

// ---- WP-6: two-input join differential -------------------------------------
// A join oracle is `(probe, build, JoinQuery) -> ResultSet`, so the identical
// runner + comparator serve both backends (independent reference / DuckDB).
using JoinOracleFn =
    std::function<ResultSet(const Table&, const Table&, const JoinQuery&)>;

// Run engine-join-vs-oracle and return the verdict. batch_size is forwarded to
// both Scans (varying it exercises build-across-batches and the output-batch
// fan-out tail).
DiffResult run_join_differential(const Table& probe, const Table& build,
                                 const JoinQuery& jq, const JoinOracleFn& oracle,
                                 std::size_t batch_size = Scan::kDefaultBatchSize);

// Convenience: engine join vs the independent reference oracle.
DiffResult run_join_vs_reference(
    const Table& probe, const Table& build, const JoinQuery& jq,
    std::size_t batch_size = Scan::kDefaultBatchSize);

}  // namespace qe::oracle
