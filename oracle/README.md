# oracle/

The differential-testing oracle: the engine's results are compared against a
golden model on seeded random queries, on both ARM/NEON and x86/AVX-512.

**Status (WP-3): seeded (this is the WP-9 seed).** The harness, comparator,
generators, and the first mutation self-test are stood up here and run in CI.

Pieces:
- **`result_set.{h,cpp}`** — the oracle-agnostic `ResultSet` (typed, nullable
  cells) + the comparison contract: **D12** canonicalization (sort both sides on
  all output columns when there is no ORDER BY) and **D11** float tolerance
  (integer/exact columns exact; F64 with relative+absolute epsilon). Also
  `drain_operator()` (engine tree → ResultSet).
- **`logical_query.{h,cpp}`** — one `LogicalQuery` (optional BOOL filter + named
  projections) that BOTH backends execute; `build_engine_pipeline()` assembles
  the `scan → [filter] → project` operator tree.
- **`reference_oracle.{h,cpp}`** — an **independent** backend: computes the query
  over the whole table in one shot (no operators/batching), so "engine ==
  reference" meaningfully checks the operator pipeline. Always available.
- **`sql_render.{h,cpp}`** — renders a `LogicalQuery` to SQL (no DuckDB
  dependency; unit-testable).
- **`duckdb_oracle.{h,cpp}`** — the **authoritative** DuckDB backend (D16). Built
  only when the amalgamation is staged (`third_party/duckdb/`, see its
  `VENDORING.md`); feeds the identical comparator.
- **`generators.{h,cpp}`** — seeded schema/data/query generation, constrained so
  the engine/DuckDB **divergence** cases never arise (see the header).
- **`differential.{h,cpp}`** — the runner: engine vs a pluggable oracle.

DuckDB / SQLite may appear **only** here and under `tests/` (the from-scratch gate
exempts both). DuckDB is never linked into an engine target.
