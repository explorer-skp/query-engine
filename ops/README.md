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

Added in WP-4 (the shared hash-table substrate, **`hashtable.h` FROZEN**):
- **`hashtable.{h,cpp}`** — open-addressing, linear-probe, power-of-two table
  (D8) shared by WP-5 (group-by) and WP-6 (join). `insert_or_find` returns a
  stable, dense group id per distinct key (build/group path); `find` probes
  without inserting (join probe path). Single or composite keys over
  I32/I64/F64/BOOL/TS. F64 keys canonicalize -0.0→+0.0 and all-NaN→one NaN.
  NULL semantics are caller-chosen via `NullPolicy` (`kEqual` = GROUP BY:
  NULLs group together; `kNeverMatch` = join: NULLs never match) — neither baked
  in. Group ids are stable across growth/rehash. Load factor + initial capacity
  are named config (`HashTableConfig`), power-of-two enforced at runtime; nothing
  hardcoded.
- **`hash_kernels.{h,cpp}` + `hash_scalar.cpp`** — the from-scratch mixing hash
  (splitmix64 finalizer) as a vector/scalar twin (Highway behind `_vec`;
  independent scalar in `_scalar.cpp`). The probe/slot walk is shared scalar
  control flow (inherently sequential); the data-parallel hashing is the
  vectorized part. `HashPath::{kVector,kScalar}` drives the scalar==vector
  differential end-to-end.
- **`hash_internal.{h,cpp}`** — non-frozen seam: key normalization + per-batch
  hashing + key equality, shared by the real table and the test-only mutant so
  the mutant differs by exactly one probe/grow step.
- **`hashtable_mutants.{h,cpp}`** — TEST-ONLY planted-probe mutants (skipped
  rehash on growth; probe stops one slot early). Not linked into any engine
  target; the mutation self-test shows the reference cross-check catches them.

From-scratch: no query-engine/dataframe libraries here (the gate covers `ops/`).
The DuckDB differential that validates these operators lives under `oracle/`.
WP-4 is a substrate, validated by standalone unit + fuzz + scalar==vector +
mutation self-tests (not yet the DuckDB oracle; WP-5/WP-6 wire it in there).
