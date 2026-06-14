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

// WP-6: a generated two-input join case — two correlated Tables (probe/build) and
// the JoinQuery to run over them. The two tables share the key column types (keys
// are columns 0..nk-1 on BOTH sides); the build side draws keys from a small
// shared DOMAIN (so distinct keys fan out and skew), and the probe side draws
// in-domain (match) or out-of-domain (no-match) keys at a varied rate, with a
// varied fraction of NULL keys on each side (kNeverMatch). The seed is printed by
// the harness for replay.
struct JoinCase {
    Table probe;
    Table build;
    JoinQuery query;
};

// Generate one random join case. Varies: #keys (1..2), key types, key cardinality
// & skew (hot key), probe match rate, NULL-key fraction, row counts, payload
// columns, and INNER vs LEFT.
JoinCase gen_join_case(std::mt19937_64& rng);


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

// Generate a GROUP BY LogicalQuery (WP-5): an optional BOOL filter, a random key
// subset (0..3 distinct columns; 0 => a single global aggregate), and a random
// aggregate subset (1..4 of COUNT(*)/COUNT/SUM/MIN/MAX/AVG). SUM/AVG are emitted
// only over numeric columns, and the data magnitude bounds above keep every
// per-group integer sum provably within the I64 accumulator (the SUM result type;
// see ops/aggregate.h and the WP-5 report), so DuckDB's HUGEINT SUM, cast back to
// BIGINT in the SQL, never disagrees.
LogicalQuery gen_group_by_query(std::mt19937_64& rng, const Schema& schema);

// Generate an ORDER BY LogicalQuery (WP-7): an optional BOOL filter, a projection
// of EVERY column as-is (so the output schema spans all types), and an ORDER BY
// over a random permutation of those output columns. The first 1..ncols keys are
// "interesting" (random ASC/DESC + NULLS FIRST/LAST); the REMAINING columns are
// appended as deterministic tiebreakers (ASC NULLS LAST). Appending the rest makes
// the sort a TOTAL order, so the POSITIONAL differential (D12) is unambiguous:
// remaining ties are only between rows equal in every column (element-wise equal),
// whose relative order is therefore unobservable. The existing data generators
// emit finite, non-NaN, non-(-0.0) floats (see above), so F64 ordering is fully
// determined and matches DuckDB. See the WP-7 report on float/NaN determinism.
LogicalQuery gen_order_by_query(std::mt19937_64& rng, const Schema& schema);

}  // namespace qe::oracle
