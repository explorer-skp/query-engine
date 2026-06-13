# ops/

Pull-based vectorized operators implementing the `Operator` contract
(`open` / `next` / `close` / `output_schema`). Operators compose into the execution
pipeline; each `next()` returns one ~2048-value batch.

**WP-0 status:** DRAFT interface stub only (`operator.h`). Frozen at WP-3.
No operator is implemented yet. From-scratch: no query-engine/dataframe libraries here.
