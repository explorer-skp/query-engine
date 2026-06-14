//  WP-3 (seed of WP-9): the oracle-agnostic RESULT SET — a materialized table of
//  typed, nullable cells — plus the comparison that decides engine == oracle.
//  Both sides (the engine's operator tree and whatever oracle: the independent
//  reference, or DuckDB) produce a ResultSet; compare_result_sets() applies the
//  frozen comparison contract:
//
//   * D12 canonicalization: with no explicit ORDER BY the row order is undefined,
//     so BOTH result sets are sorted on all output columns before diffing
//     (UNORDERED mode, the default — what every non-sort test relies on). With an
//     explicit ORDER BY (WP-7) the query DEFINES the row order, so the comparator
//     runs in ORDERED mode: NO canonicalization, rows compared positionally in
//     emitted order. Select it with the `ordered` flag on compare_result_sets().
//   * D11 float tolerance: integer-family columns (I32/I64/BOOL/TS) compare by
//     EXACT equality; F64 columns compare with relative+absolute epsilon
//     (summation order etc. make bit-exact float impossible in general). The
//     chosen epsilons are kAbsEps / kRelEps below.
//   * Nullness compares exactly on every cell (a NULL only matches a NULL).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/column.h"
#include "core/types.h"
#include "ops/operator.h"

namespace qe::oracle {

// One output value. For I32/I64/BOOL/TS the payload is `i`; for F64 it is `f`.
struct Cell {
    bool is_null = false;
    std::int64_t i = 0;
    double f = 0.0;
};

// A materialized result: column types + rows of cells (row-major).
struct ResultSet {
    std::vector<Type> types;
    std::vector<std::vector<Cell>> rows;

    std::size_t num_cols() const { return types.size(); }
    std::size_t num_rows() const { return rows.size(); }
};

// Float comparison epsilons (D11). Absolute guards values near zero; relative
// guards large magnitudes. A pair (a,b) is "equal" iff
//   |a-b| <= kAbsEps + kRelEps * max(|a|,|b|).
inline constexpr double kAbsEps = 1e-9;
inline constexpr double kRelEps = 1e-9;

// Drain a fully-built operator tree (already constructed; this calls open(),
// pulls every batch, and close()) into a ResultSet. Respects each batch's
// optional selection vector and the validity precedence.
ResultSet drain_operator(Operator& root);

// Compare two result sets per D11/D12. `equal` is the verdict; `message` is a
// human-readable explanation of the FIRST difference found (empty on equal).
struct DiffResult {
    bool equal = false;
    std::string message;
};

// `ordered == false` (default): UNORDERED — canonicalize both sides (sort on all
// output columns) before diffing; the historical WP-3..WP-5 behavior. `ordered ==
// true`: ORDERED — the query carries an explicit ORDER BY (WP-7), so rows are
// compared POSITIONALLY in emitted order with NO canonicalization. Cell-level
// rules (D11 float epsilon, exact integers, exact nullness) are identical in both
// modes.
DiffResult compare_result_sets(const ResultSet& engine, const ResultSet& oracle,
                               bool ordered = false);

// Render a result set (canonicalized) to a compact string — for failure
// diagnostics. Caps the number of rows printed.
std::string to_debug_string(const ResultSet& rs, std::size_t max_rows = 16);

}  // namespace qe::oracle
