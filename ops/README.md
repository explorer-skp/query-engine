# ops/

Pull-based vectorized operators implementing the `Operator` contract
(`open` / `next -> optional<Batch>` / `close` / `output_schema`). Operators compose
into the execution pipeline; each `next()` returns one ≤2048-row batch.

**Status (WP-3): `operator.h` FROZEN.** The base `Operator` interface is the
universal substrate for all later operators; do not add to it without an ICR.

Implemented this WP:
- **`table.{h,cpp}`** — the in-memory columnar source: a `Schema` + one
  `OwnedColumn` per field (WP-1 owning layer). Owned by WP-3; WP-8's plan layer
  will drive its construction later.
- **`scan.{h,cpp}`** — leaf operator; yields the Table as dense ≤`batch_size`-row
  Batches whose Column views point **zero-copy** into the Table. `batch_size`
  defaults to 2048 (D4) and must be a multiple of 64 so a batch's validity
  sub-view is a whole-word pointer offset (validity is 64-values-per-word).
- **`filter.{h,cpp}`** — evaluates a BOOL predicate via `expr::evaluate` and
  **compacts** the passing rows (D6 first cut). NULL predicate ⇒ row does not
  pass. Correctly handles an input batch that already carries a selection vector
  (evaluate applies it; compaction is out-of-place, dodging the §12 aliasing
  hazard). Never emits an empty batch.
- **`project.{h,cpp}`** — evaluates a list of named expr trees per batch into a
  fresh dense `OwnedBatch` (the SELECT list).

From-scratch: no query-engine/dataframe libraries here (the gate covers `ops/`).
The DuckDB differential that validates these operators lives under `oracle/`.
