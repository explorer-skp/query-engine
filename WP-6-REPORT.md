# WP-6 Report — Hash equi-join (`ops/join.{h,cpp}`)

Scope: a pull-based hash equi-join operator (INNER + LEFT-OUTER, single & composite
keys) built on the frozen `HashTable` substrate (`ops/hashtable.h`), wired into the
DuckDB differential oracle. No frozen interface was edited. **No ICRs.**

## Public surface (`ops/join.h`) — what WP-8 consumes

```cpp
enum class JoinType  { Inner, Left };
enum class GatherPath { kVector, kScalar };   // test seam for the build-row gather

class HashJoin : public Operator {
  static constexpr std::size_t kOutBatch = 2048;
  HashJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
           std::vector<std::uint32_t> probe_keys,
           std::vector<std::uint32_t> build_keys, JoinType type);
  void set_paths(HashPath, GatherPath);       // before open(); WP-8 ignores
  // open()/next()/close()/output_schema() — the frozen Operator contract.
};
```

Small and explicit, mirroring `AggSpec`-style design: two children + two parallel
key-index lists + a `JoinType`. Probe is the **left** table (the one LEFT-OUTER
preserves). `set_paths` is a test-only seam (defaulted to vector) so scalar==vector
is a true end-to-end check.

### Output column order (documented contract)
**All probe columns (child order) followed by all build columns (child order).**
Types = probe types ++ build types. Field names are the children's names verbatim
and **may collide** across sides (both `c0`, …) — consumers identify columns by
**position**, never name. The SQL renderer aliases output columns `o0..oN` by
position for exactly this reason, so the engine/DuckDB/reference all agree on order
without depending on names.

## Build-side multiplicity map design
The frozen table **dedups distinct keys to one stable group id and stores no
payloads**, so collecting matches per key is this WP's job:

* `open()` drains the build child fully. Each batch's key columns are inserted via
  `HashTable::insert_or_find` with **`NullPolicy::kNeverMatch`** → each row gets a
  group id (null-key rows get fresh "dead", unfindable ids — the join-NULL
  semantics, validated against DuckDB).
* All build columns are materialized **densely** into a column-major raw-byte
  `BuildStore` (`ops/join_internal.h`), one store row per drained build row, with a
  per-row validity flag per column. (Build batch views are transient; the join must
  own the values it later gathers.)
* The map is `group_id -> std::vector<build_row_index>` (`group_rows`). For row `k`
  with group `g`, we append the value row to the store and push its index onto
  `group_rows[g]`. Dead (null-key) groups simply collect their one unfindable row
  and are never probed.

## Probe / emit
`next()` pulls a probe batch and expands it into parallel pair arrays
`(probe_phys_row, build_row)`:
* matched probe row (`find()` ≠ `kNoGroup`): one pair per build row in
  `group_rows[g]` (the gather fan-out);
* INNER + unmatched: nothing;
* LEFT + unmatched: one pair `(phys, kNullBuildRow)`.

Pairs are emitted in dense `≤2048`-row batches across successive `next()` calls,
so a **single high-fanout probe row or a LEFT unmatched row is split at row
granularity** and the output-batch tail is always exact (the planted-bug hotspot —
covered by a 3000-row fan-out edge test and the `kFanoutTailOffByOne` mutant). The
live probe batch is held in `cur_probe_` and only re-pulled once its pairs are
drained, so its zero-copy column views stay valid throughout emit.

## Gather reuse (scalar twin)
The probe-column and build-column materialization reuse the **WP-1 gather kernels**
(`simd/gather_kernels.h`): `gather32` for I32, `gather64` for I64/F64/TS, and the
scalar-only `gather8` for BOOL (no vector twin — documented in the header). Both the
Highway `_vec` path and its independent `_scalar` twin are driven by `GatherPath`,
exercised in `join_scalar_vector_test` directly (random src/idx) **and** end-to-end
(every `HashPath × GatherPath` combination produces an identical ResultSet). Build
sentinel rows (`kNullBuildRow`) gather a safe index 0 then get `set_null`, so a
sentinel never drives an out-of-range read; gather `out`/`src` never alias (out is a
fresh OwnedColumn), so the §12 aliasing hazard cannot arise.

## NULL / LEFT / float semantics (matched to DuckDB)
* A key tuple with **any NULL matches nothing**, including NULL==NULL, on either
  side (`kNeverMatch`). In SQL this is exactly `JOIN ... ON p.k = b.k` (a NULL
  operand makes `=` NULL → non-match). Validated.
* INNER: only matching pairs (k build matches ⇒ k rows). LEFT: every probe row
  appears ≥ once; unmatched ⇒ build columns all NULL.
* F64 key equality uses the table's canonicalization (−0.0==+0.0, NaN==NaN); the
  reference oracle reproduces it independently and DuckDB's `=` agrees on the
  (finite, NaN-free) keys the generators emit.

## Oracle extensions (the WP-9 scaffold; not frozen)
* `oracle/logical_query.{h,cpp}` — `JoinQuery` (two tables + per-side keys + type),
  `join_output_schema`, `build_join_pipeline` (single source of truth: scan(probe) +
  scan(build) → HashJoin).
* `oracle/sql_render.{h,cpp}` — `join_sql`: `SELECT CAST(p.<c> AS T) AS o0, … FROM
  probe p [LEFT] JOIN build b ON p.k = b.k [AND …]`, columns aliased by position,
  each CAST to the engine type (the WP-5 CAST precedent).
* `oracle/reference_oracle.{h,cpp}` — `run_join_reference`: an **independent**
  `std::map`-indexed nested scan sharing no code with HashJoin/HashTable/gather.
* `oracle/duckdb_oracle.{h,cpp}` — `run_join_duckdb`: loads `probe`+`build`, runs the
  rendered join, reads back. Authoritative (D16). Generalized the loader to a named
  table and factored a shared `read_result`.
* `oracle/generators.{h,cpp}` — `gen_join_case`: two correlated tables sharing key
  types, varying #keys (1–2), key cardinality & skew (hot key), probe match rate,
  NULL-key fraction, row counts, payload columns, INNER/LEFT. Out-of-domain probe
  keys `[1000,2000]` are disjoint from the build domain `[0,99]` so a non-matching
  key provably misses.
* `oracle/differential.{h,cpp}` — `run_join_differential` / `run_join_vs_reference`.
  The join is UNORDERED, so it uses the historical D12 canonicalization
  (`compare_result_sets(..., ordered=false)` — the default) — no positional compare
  (that is WP-7's concern; left untouched).

## Output column order / why probe-then-build
Probe-then-build is the natural pull order (probe streams, build is the lookup side)
and lets WP-8 address probe columns at stable low indices. Names may collide, which
is why the contract is positional and the SQL renderer aliases by ordinal.

## Mutation self-test (`ops/join_mutants.*`, `tests/join_mutation_test.cpp`)
Faithful copy of the build / pair-gen / output loops reusing the **same**
`join_internal.h` helpers (BuildStore + gather emitters), one planted defect each:
* `kDropProbeMatch` (the §5 named bug) — skip one matched row → row-count short.
* `kLeftWrongNull` — LEFT unmatched row emits build row 0's real values, not NULL.
* `kCompositeFirstKeyOnly` — match composite keys on the first column only → over-join.
* `kFanoutTailOffByOne` — final output batch one row short (cursor still advances, so
  the row is dropped for good, not deferred).

All four are CAUGHT by the differential; the real operator PASSES the identical diff.

## Assumptions
* Probe and build key columns are positionally type-matched (asserted; generators
  guarantee it). Equi-join only (`=`); ≥ 1 key.
* Empty build (INNER ⇒ 0 rows; LEFT ⇒ all-NULL build) and empty probe (both ⇒ 0
  rows) handled without touching `compact_column` (the operator builds OwnedColumns
  and gathers directly; the known `compact_column` n==0 abort is never reached).

## Cross-WP integration note for the final review
WP-7 (sort) is extending the **same** non-frozen oracle scaffold concurrently. The
merged tree is coherent and the from-scratch grep is clean. One transient hazard
observed: at points the shared `CMakeLists.txt` referenced WP-7 source files
(`tests/sort_*`, `bench/bench_sort_main.cpp`) that did not yet exist, which breaks a
**fresh** `cmake` configure for *any* worker until those files land. It self-resolved
as WP-7 created the files; no action needed, but worth knowing if a clean configure
fails mid-merge. I did not edit any WP-7 file. WP-7 added an `ordered` parameter
(defaulted) to `compare_result_sets`; my join calls use the default (UNORDERED),
which is correct for joins.

## Definition-of-done checklist
- Conforms to frozen `Operator`; `ops/join.h` is the WP-6 public surface. No frozen
  header edited. No ICR.
- Scalar twin: build-row gather reuses `simd/gather_kernels` (`_vec`) + its `_scalar`
  twin, exercised directly and end-to-end across `HashPath × GatherPath`.
- Oracle: INNER + LEFT diffed vs the independent reference **and** authoritative
  DuckDB on seeded random data + edge cases (single/composite key, no-match,
  all-match, skew, nulls-in-key both sides, empty build, empty probe, high-fanout
  >2048 tail). Canonicalized per D12; F64 per D11.
- ≥1 must-flag mutant (four planted); suite shown to flag each and pass on the real op.
- Determinism: every randomized test seeded; seed printed; `--seed N` replays.
- ASan + UBSan green (re-run; commands below).
- From-scratch grep clean over `core simd expr ops plan tsx`.

---

## Commands

Build (release, DuckDB auto-enabled since the amalgamation is staged):
```
cmake --preset release
cmake --build build --target join_scalar_vector_test join_differential_test join_mutation_test -j8
```

Run the join differential vs BOTH reference and DuckDB (authoritative):
```
./build/join_differential_test --seed 20260614      # CI seed
./build/join_differential_test                       # fuzz mode; prints a fresh seed
# replay a failure:  ./build/join_differential_test --seed <printed>
```

scalar==vector (kernels + end-to-end across HashPath × GatherPath):
```
./build/join_scalar_vector_test --seed 20260614
```

Mutation self-test (suite FLAGS each mutant; real op passes):
```
./build/join_mutation_test --seed 20260614
# observed:
#   kDropProbeMatch caught: row count mismatch: engine=2 oracle=3
#   kLeftWrongNull caught: null mismatch at canonical row 3 col 1: engine=1 oracle=NULL
#   kCompositeFirstKeyOnly caught: row count mismatch: engine=2 oracle=1
#   kFanoutTailOffByOne caught: row count mismatch: engine=2 oracle=3
#   test cases: 5 | 5 passed   (the REAL join passes the identical diff)
```

ASan + UBSan gate (re-run, all green):
```
cmake --preset asan
cmake --build build-asan --target join_scalar_vector_test join_differential_test join_mutation_test -j8
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./build-asan/join_scalar_vector_test --seed 20260614
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./build-asan/join_differential_test  --seed 20260614
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./build-asan/join_mutation_test      --seed 20260614
```

From-scratch gate:
```
bash scripts/check_forbidden_includes.sh     # clean
```

No performance numbers are claimed in this WP (correctness WP; the vec-vs-scalar
ratio is WP-7's benchmark deliverable). All measurements would be tagged host+isa
per RIGOR.md if cited.
