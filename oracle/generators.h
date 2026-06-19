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
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/owned_batch.h"
#include "oracle/logical_query.h"
#include "ops/table.h"
#include "plan/plan.h"
#include "tsx/asof.h"    // WP-12: AsofType / AsofCase
#include "tsx/window.h"  // WP-13: WindowMode / WindowCase

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

// WP-12: a generated backward AS-OF join case — two correlated tick-like Tables
// plus the partition keys / timestamp columns / type / optional tolerance to run
// over them. Layout (both sides): columns 0..nk-1 are the shared partition keys
// (nk in 0..2 — global / single / composite), column nk is the TIMESTAMP (TS/I32/
// I64), then 0..2 payload columns. The generator exercises the §5 hazards:
//   * irregular timestamps with gaps (probe before any build row => no-match);
//   * EXACT-timestamp ties at the boundary (some probe ts == a build ts, so the
//     `>=` boundary is hit head-on);
//   * single & composite & zero keys; INNER & LEFT; empty sides.
// NULL KEYS and NULL TIMESTAMPS are deliberately NOT generated — two verified
// DuckDB v1.1.3 ASOF divergences (a NULL probe ts matches a NULL build ts; NULL-key
// matching is data-dependent) make a byte-for-byte diff impossible/flaky, so (per
// the generators.h divergence philosophy) we constrain generation and validate the
// engine's principled "NULL never matches" semantics against the independent
// reference instead (see the asof_differential_test NULL edge cases + the .cpp note).
// DETERMINISM: every build-side timestamp is GLOBALLY DISTINCT, so build (key,
// timestamp) is unique and the nearest-preceding pick is unambiguous across engine
// / reference / DuckDB (no flaky tie). The seed is printed by the harness for
// replay. Tables live by value (the test wraps them in stable storage for the
// plan's borrowing Scan nodes).
struct AsofCase {
    Table probe;
    Table build;
    std::vector<std::uint32_t> left_keys;
    std::vector<std::uint32_t> right_keys;
    std::uint32_t left_time = 0;
    std::uint32_t right_time = 0;
    qe::tsx::AsofType type = qe::tsx::AsofType::Inner;
    std::optional<std::int64_t> tolerance;  // unset => unbounded
};

AsofCase gen_asof_case(std::mt19937_64& rng);

// WP-13: a generated windowed-aggregation case — one tick-like Table plus the
// window mode / partition keys / timestamp column / param (bucket width W for
// Tumbling, PRECEDING row count P for Sliding) / aggregate subset to run over it.
// Layout: columns 0..nk-1 are the shared partition keys (nk in 0..2 — global /
// single / composite), column nk is the TIMESTAMP (TS/I32/I64), then 0..2 payload
// columns. The generator exercises the §5 hazards: occupied/empty buckets, single &
// composite & zero keys, frames spanning batch boundaries incl. a tail, empty input,
// all-null frame slices, and NULL partition keys (kEqual: group together). The
// aggregate subset is overflow-safe (per-bucket / per-frame SUM provably within the
// I64 accumulator). DETERMINISM / the two GENERATION CONSTRAINTS: every timestamp is
// >= 0 (Tumbling: integer bucketing is divergence-free vs DuckDB) AND globally
// distinct (Sliding: the window's ORDER BY t is a TOTAL order per partition, so every
// running value is deterministic). Timestamps are never NULL. The seed is printed by
// the harness for replay. The Table lives by value (the test wraps it in stable
// storage for the plan's borrowing Scan node).
struct WindowCase {
    Table input;
    qe::tsx::WindowMode mode = qe::tsx::WindowMode::Tumbling;
    std::vector<std::uint32_t> keys;
    std::uint32_t time = 0;
    std::int64_t param = 1;  // W (tumbling) / P (sliding)
    std::vector<AggSpec> aggs;
};

WindowCase gen_window_case(std::mt19937_64& rng);


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

// WP-7b: a dictionary-encoded VARCHAR (Type::STR) column of `n` rows, each a random
// pick from `alphabet` (a small set of short ASCII strings, no embedded NULs),
// interned into a FRESH per-column dict, with ~null_pct% nulls. Two columns built
// from the same alphabet share the string VALUES but generally get DIFFERENT codes
// (first-use order differs) — the cross-dictionary case STR equality/join/group/sort
// must resolve by VALUE, never by raw code.
OwnedColumn gen_string_column(std::mt19937_64& rng,
                              const std::vector<std::string>& alphabet,
                              std::size_t n, int null_pct);

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

// WP-8: a generated DEEP composite PLAN case — a pipeline no single earlier WP
// exercises: scan(probe) -> [filter] -> join(scan(build)) -> aggregate -> sort.
// Built on gen_join_case (two correlated tables sharing key column(s)); the
// aggregates are overflow-proof (COUNT/MIN/MAX) and the final ORDER BY spans every
// output column (a TOTAL order over rows distinct by group key), so the positional
// differential is unambiguous and DuckDB-exact. The Tables live in `tables` as
// std::unique_ptr (STABLE addresses) because the Plan's Scan nodes BORROW them:
// moving the PlanCase must not dangle those pointers. The seed is printed by the
// harness for replay.
struct PlanCase {
    std::vector<std::unique_ptr<Table>> tables;  // stable storage for Scan borrows
    qe::plan::Plan plan;
};

PlanCase gen_plan_case(std::mt19937_64& rng);

}  // namespace qe::oracle
