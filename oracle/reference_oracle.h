//  WP-3 (seed of WP-9): the INDEPENDENT REFERENCE oracle. It computes a
//  LogicalQuery's result over the whole Table in ONE shot — no Scan/Filter/
//  Project operators, no batching, no compaction, no per-batch selection vectors
//  — and returns a ResultSet. It deliberately shares NO code path with the
//  operator pipeline it is checking, so "engine == reference" is a meaningful
//  differential of the WP-3 operators (batching, tail handling, compaction,
//  selection-vector flow) against a monolithic computation.
//
//  It DOES reuse expr::evaluate() (the WP-2 frozen, separately oracle-bound
//  expression engine): WP-3's new surface is the operator assembly, not
//  expression semantics, so evaluating the predicate/projections over the whole
//  table at once is the right reference. The DuckDB backend (oracle/sql_oracle.h,
//  staged separately) is the ultimate cross-check of expression semantics; it
//  feeds the IDENTICAL comparator (oracle/result_set.h).
#pragma once

#include "oracle/logical_query.h"
#include "oracle/result_set.h"
#include "ops/table.h"

namespace qe::oracle {

ResultSet run_reference(const Table& table, const LogicalQuery& q);

}  // namespace qe::oracle
