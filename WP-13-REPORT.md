# WP-13 Report — Windowed / time-bucketed aggregation

*(Reconstructed during the 2026-07-05 full-codebase audit — the WP shipped and
was accepted in the M4 integration commit d1aff62, but its report was never
written, a definition-of-done gap the audit flagged. Design notes below are
drawn from the accepted code + spec + project change log.)*

## What shipped

One operator, `qe::tsx::Window` (frozen surface `tsx/window.h`), carrying both
modes:

- **TUMBLING** — time-bucketed GROUP BY. `bucket_start = (t / W) * W` with C++
  integer division, i.e. truncation toward zero — this equals floor bucketing
  only for `t >= 0`, which the oracle grammar guarantees; the rendered SQL
  `(t - t % W)` is algebraically identical under truncation, so even negative
  t would agree with the *rendered* query (the divergence is vs floor-based
  `time_bucket` semantics only). Grouping via the frozen HashTable
  (NullPolicy::kEqual; NULL timestamps form their own bucket) + the shared
  `ops/agg_internal.h` scatter.
- **SLIDING** — running aggregate over `ROWS BETWEEN P PRECEDING AND CURRENT
  ROW`, per partition, ordered by `(keys…, t)` via the frozen Sort with
  `NullOrder::Last`. Running int SUM/COUNT; monotonic deques for MIN/MAX.

## Deviations from the spec (deliberate, code-documented)

- **Float SUM/AVG is O(n·P), not O(n)**: the spec's "running aggregate in
  O(n)" is implemented for the integer family only. Incremental float
  subtraction drifts past the D11 epsilons near cancellation, so float SUM/AVG
  recompute the frame slice in time order — bit-stable vs the reference. P <= 50
  in the grammar, so the cost is bounded. (window.cpp header documents this.)
- The one vector/scalar twin is the sliding child-gather (GatherPath);
  bucket-scatter and the ring/deque sweep are scalar (WP-5/WP-6 precedent).

## Audit-driven fixes folded in (2026-07-05)

- **NaN policy (audit C2)**: MIN/MAX now use the NaN-greatest total order
  (matches DuckDB and the Phase-1 Aggregate) in BOTH modes; pre-fix, tumbling
  emitted ±inf for all-NaN buckets and the sliding deques were order-dependent.
  Covered by a hand differential case vs DuckDB (window_differential_test
  "NaN values in F64 MIN/MAX") and the catalog mutant `agg_max_drops_nan`.
- **STR guards (audit H5)**: STR partition keys / aggregate inputs and
  non-integer time columns are rejected with real throws in every build
  (previously assert-only; under NDEBUG a STR partition key silently merged
  all partitions).
- **NULL-timestamp sliding windows (audit H1/H2)**: the reference now sorts
  NULL ts LAST (was: as t=0) and the rendered OVER clause says ASC NULLS LAST
  explicitly; covered by a new differential case vs DuckDB.

## Validation

- `window_differential_test` — engine vs independent reference (std::map /
  brute-force frame scan; shares no engine code) vs real DuckDB
  (GROUP BY / OVER renders), seeded, batch sizes {64,256,2048}.
- `window_scalar_vector_test` — sliding gather twin gate.
- `window_mutation_test` + catalog entries `window_frame_off_by_one` (and the
  audit's `agg_max_drops_nan` for the shared fold) — all flagged.

## Reproduce

```bash
cmake --preset release && cmake --build build -j
./build/window_differential_test --seed 20260614
./build/window_scalar_vector_test --seed 20260614
./build/window_mutation_test --seed 20260614
```
