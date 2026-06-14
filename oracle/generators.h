//  WP-3 (seed of WP-9): seeded generators for SCHEMA, DATA, and QUERY. Everything
//  is driven by one std::mt19937_64 whose seed the test harness prints, so any
//  failure replays from a single `--seed N` command (D13 / RIGOR.md rule 5).
//
//  ORACLE-DIVERGENCE HANDLING (critical — see expr/expr.h and the WP report).
//  This engine intentionally diverges from DuckDB on a handful of cases where
//  DuckDB RAISES but the engine yields NULL: div/mod-by-zero, out-of-range or
//  non-finite float->int casts, and signed-integer overflow on +,-,*. We pick
//  option (a) from the brief: CONSTRAIN GENERATION so these cases never arise, so
//  DuckDB never raises and there is nothing to reconcile. Concretely:
//    * Integer data and arithmetic are magnitude-bounded so +,-,* cannot overflow
//      int32/int64 (kI32Abs, kI64Abs, multipliers <= kMulAbs, depth-limited).
//    * No division/modulo is generated (also dodges DuckDB's `/` float-division
//      vs `//` integer-division mismatch). The engine and expr layer SUPPORT
//      them; they are simply out of the first oracle's grammar.
//    * Casts are WIDENING only (I32->I64, I32->F64, I64->F64), never lossy/OOR.
//    * F64 data is finite and moderate (no inf/NaN), so no float->int OOR and no
//      NaN canonicalization ambiguity.
//  The SQL backend (oracle/sql_oracle.*) additionally treats any DuckDB
//  statement error as "regenerate, don't diff" — a defensive backstop that should
//  never fire given the constraints above.
#pragma once

#include <cstddef>
#include <random>

#include "oracle/logical_query.h"
#include "ops/table.h"

namespace qe::oracle {

// Magnitude bounds that keep all generated integer arithmetic overflow-free.
inline constexpr std::int32_t kI32Abs = 30000;
inline constexpr std::int64_t kI64Abs = 1'000'000'000LL;
inline constexpr std::int64_t kMulAbs = 100;  // multiplier literals
inline constexpr double kF64Abs = 1'000'000.0;

struct GenConfig {
    std::size_t min_rows = 0;
    std::size_t max_rows = 5000;  // spans several 2048 batches incl. a tail
    int min_cols = 2;
    int max_cols = 5;
    int null_pct = 15;
};

// Generate a random schema (column 0 is always numeric so predicates/projections
// always have a numeric operand). Names are "c0","c1",...
Schema gen_schema(std::mt19937_64& rng, const GenConfig& cfg = {});

// Generate a Table of random data conforming to `schema` (bounded magnitudes,
// finite floats, ~null_pct% nulls).
Table gen_table(std::mt19937_64& rng, const Schema& schema,
                const GenConfig& cfg = {});

// Generate a LogicalQuery (optional BOOL filter + 1..4 named projections) over
// `schema`, within the divergence-safe grammar described above.
LogicalQuery gen_query(std::mt19937_64& rng, const Schema& schema);

}  // namespace qe::oracle
