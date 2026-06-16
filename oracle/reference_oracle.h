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

#include <cstdint>
#include <optional>
#include <vector>

#include "oracle/logical_query.h"
#include "oracle/result_set.h"
#include "ops/table.h"
#include "plan/plan.h"
#include "tsx/asof.h"  // WP-12: AsofType

namespace qe::oracle {

ResultSet run_reference(const Table& table, const LogicalQuery& q);

// WP-8: the INDEPENDENT reference for an arbitrary plan::Plan tree. It walks the
// plan bottom-up, MATERIALIZING each node's output as a Table and computing that
// node with the EXISTING per-node references (run_reference for
// scan/filter/project/group-by/order-by, run_join_reference for join). It shares
// NO code path with the engine's operator tree (plan.lower()), so "engine ==
// reference" is a meaningful differential of the WHOLE composition (alongside the
// authoritative DuckDB plan diff). The borrowed Table(s) the plan's Scan nodes
// reference must outlive this call.
ResultSet run_plan_reference(const qe::plan::Plan& p);

// WP-6: the INDEPENDENT reference for a two-input equi-join. Computes the join
// DIRECTLY with a std::map keyed by the canonical build-key tuple plus a nested
// scan — it shares NO code path with the engine's HashJoin / HashTable / gather,
// so "engine == reference" is a meaningful differential (alongside the
// authoritative DuckDB diff). Reproduces the documented key semantics
// independently: a key tuple with ANY NULL never matches (kNeverMatch), F64 zero
// canonicalizes to +0.0 and NaN to a canonical quiet NaN so equality matches the
// engine's; output column order is probe columns then build columns.
ResultSet run_join_reference(const Table& probe, const Table& build,
                             const JoinQuery& jq);

// WP-12: the INDEPENDENT reference for a backward AS-OF join. For each probe row it
// SCANS the whole build side (brute force) for rows with an equal, non-NULL key
// tuple and a non-NULL timestamp tb <= the probe timestamp, picks the GREATEST such
// tb (nearest preceding), and applies the optional tolerance (drop if t - tb >
// tolerance). It shares NO code path with the engine's Sort / HashTable / merge /
// gather, so "engine == reference" is a meaningful differential (alongside the
// authoritative DuckDB ASOF JOIN diff). It reproduces the documented semantics
// independently: a NULL in any key or in either timestamp never matches; F64 key
// zero canonicalizes to +0.0 and NaN to a canonical quiet NaN; output column order
// is probe columns then build columns; INNER drops unmatched probe rows, LEFT
// NULL-fills them. (The generators keep build-side (key, timestamp) UNIQUE, so the
// nearest-preceding pick is unambiguous — see the WP report.)
ResultSet run_asof_reference(const Table& probe, const Table& build,
                             const std::vector<std::uint32_t>& left_keys,
                             const std::vector<std::uint32_t>& right_keys,
                             std::uint32_t left_time, std::uint32_t right_time,
                             qe::tsx::AsofType type,
                             std::optional<std::int64_t> tolerance);

}  // namespace qe::oracle
