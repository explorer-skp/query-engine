# WP-5 Report — Hash aggregation / GROUP BY

Scope: a pipeline-breaking hash-aggregation operator (`ops/aggregate.{h,cpp}`)
that conforms to the frozen `Operator` contract and computes grouped aggregates
over its child's output, plus its SIMD reduction kernels (+ scalar twins), a
planted-mutant operator, the oracle extensions, GROUP BY generators, and tests.

**Status: DONE.** 23/23 ctest green (incl. the WP-5 trio); the M2 GROUP BY
differential diffs green against **DuckDB** (authoritative) and the independent
reference; ASan+UBSan green; from-scratch grep clean; **no frozen header edited
(no ICR needed).**

---

## 1. Code (full files)

New (WP-5):
- `ops/aggregate.h` / `ops/aggregate.cpp` — the `Aggregate` operator.
- `ops/agg_kernels.h` — internal kernel seam (masked reductions).
- `ops/agg_kernels.cpp` — Highway **vector** path (`-Wno-error` TU).
- `ops/agg_scalar.cpp` — independent **scalar twin**.
- `ops/agg_internal.h` — non-frozen shared helpers (state cell, typed readers,
  key/agg output writers), reused by the real op and the mutant.
- `ops/aggregate_mutants.h` / `ops/aggregate_mutants.cpp` — TEST-ONLY planted-
  mutant operator (not linked into any engine target).
- `tests/agg_scalar_vector_test.cpp` — scalar==vector gate.
- `tests/aggregate_differential_test.cpp` — M2 GROUP BY oracle diff.
- `tests/agg_mutation_test.cpp` — mutation self-test.

Extended (oracle scaffold — NOT frozen, in-scope):
- `oracle/logical_query.{h,cpp}` — `GroupBy` on `LogicalQuery`; `query_output_schema`;
  `build_engine_pipeline` now emits `scan -> [filter] -> aggregate` for group-by.
- `oracle/sql_render.cpp` — renders GROUP BY + aggregates (with overflow-honest CASTs).
- `oracle/reference_oracle.cpp` — **independent** grouped-aggregate computation.
- `oracle/duckdb_oracle.cpp` — reads result columns by the shared output schema.
- `oracle/generators.{h,cpp}` — `gen_group_by_query` (random key/agg subsets).
- `CMakeLists.txt` — `qe_ops` gains the operator + kernels; `qe_aggregate_mutants`
  library; three WP-5 test targets.

Frozen headers untouched: `core/*.h`, `simd/kernel_convention.h`, `expr/expr.h`,
`ops/operator.h`, `ops/hashtable.h` (verified via `git status`).

---

## 2. Design notes

### `ops/aggregate.h` public surface (consumed later by WP-8)
- `enum class AggFunc { CountStar, Count, Sum, Min, Max, Avg }`.
- `struct AggSpec { AggFunc; uint32_t input_col; string out_name; }` with
  readable factories `count_star/count/sum/min/max/avg(...)` so call sites mirror
  the SQL. `CountStar` carries no input column.
- `Type agg_result_type(AggFunc, Type input)` — the single source of truth for
  result types/overflow policy, used by `output_schema()`, the reference, and the
  SQL renderer.
- `Aggregate(child, key_cols, aggs)` — `key_cols` are child-output column indices
  in key order; **empty = global aggregate**. Implements `open/next/close/
  output_schema` per `ops/operator.h`. `set_paths(HashPath, AggKernelPath)` is a
  test seam (defaults `kVector`) so scalar==vector is a true end-to-end check;
  WP-8 ignores it. Output schema = key columns (child name+type, key order) then
  aggregate columns (`out_name` + result type).

Rationale: the grouping side is delegated entirely to the frozen `HashTable`
(`NullPolicy::kEqual`); group ids are stable across growth, so per-group state is
just parallel arrays indexed by id.

### SUM result-type / overflow policy (the §12 candidate bug)
- `COUNT(*)/COUNT(col) -> I64`; `SUM(I32)->I64`, `SUM(I64)->I64`, `SUM(F64)->F64`;
  `AVG -> F64`; `MIN/MAX(T) -> T`.
- The integer SUM accumulator is **I64**. DuckDB's `SUM(INTEGER)/SUM(BIGINT)`
  return **HUGEINT** and never overflow. We make the diff **honest two ways at
  once**: (a) the generators bound data so every per-group integer sum provably
  fits I64 — tested envelope: ≤5000 rows × |I32|≤30000 (≤1.5e8) and |I64|≤1e9
  (≤5e12), both « 9.2e18; and (b) the rendered SQL wraps `SUM(...)` in
  `CAST(... AS BIGINT)`, so DuckDB returns BIGINT and the read-back is exact, not
  a silent I64-vs-HUGEINT disagreement. The cast never raises because the true
  sum is in range.
- **D11 epsilons** for `SUM(F64)`/`AVG`: the existing comparator's
  `kAbsEps = kRelEps = 1e-9` (`|a-b| ≤ absEps + relEps·max(|a|,|b|)`).
  Justified: ≤5000 F64 addends of magnitude ≤1e6 accumulate relative rounding
  ~`n·ε_machine ≈ 1.1e-12`, comfortably under 1e-9; summation order differs
  across engine/reference/DuckDB so bit-exactness is impossible (the D11 reason).
  Integer SUM/COUNT and MIN/MAX/key columns compare **exactly**.

### NULL / empty-group semantics (matched to DuckDB)
- `COUNT(*)` counts all rows in the group; `COUNT(col)` counts non-null.
- `SUM/MIN/MAX/AVG` ignore NULLs; a group with zero non-null inputs yields
  **NULL** (the per-group `cnt` of non-nulls drives the "seen" test). `AVG =
  sum(non-null)/count(non-null)`, DOUBLE, NULL if none.
- Global aggregate over **empty input** = exactly one row (`COUNT*=0`,
  `COUNT(col)=0`, `SUM/MIN/MAX/AVG=NULL`); GROUP BY over empty input = **zero
  rows**. Both tested vs DuckDB.
- NULL group keys group together (`kEqual`); F64 keys read back the canonical
  word (so −0.0/+0.0 collapse, NaN canonical) — the reference canonicalizes the
  same way so its output keys match the engine's.

### Output batching over >2048 groups
`open()` drains the child fully and builds all groups + state. `next()` emits
dense batches of `min(kOutBatch=2048, remaining)` groups — keys first then agg
columns — advancing a cursor until exhausted (global = one row, once). Each
output row's keys are materialized from the table's read-back
(`group_is_null`/`group_key_word`, decoded per `Type`); agg cells are finalized
directly into `OwnedColumn`s. The `key skew` test groups 6000 rows so the group
count crosses the 2048 output-batch boundary (tail batch exercised).

### Vectorization decision (honest, and the one thing worth flagging)
The genuinely data-parallel primitive in aggregation is the **contiguous masked
reduction** (sum/min/max over an identity-folded column → one accumulator). That
is implemented as a Highway vector kernel **with an independent scalar twin**
(`ops/agg_kernels.cpp` vs `ops/agg_scalar.cpp`) and is exercised in production by
the **global (zero-key) aggregate**, where the scalar==vector gate bites.

The **grouped** (≥1 key) path is an inherently-sequential **scatter** into
per-group state: duplicate group ids within a SIMD lane cannot be scatter-added
safely or portably with Highway (no portable conflict-detection primitive), so it
is scalar control flow — exactly the precedent set by the WP-4 probe walk
("inherently sequential per row … NOT vectorized"), which this brief explicitly
cites. Its correctness is pinned by the authoritative DuckDB differential and the
mutation self-test, not by scalar==vector. This is **not** an interface change and
needs no ICR; it is a candidate future optimization (e.g. AVX-512 `vpconflict`)
behind the unchanged public surface. Null-skipping in the global path lives in
the operator's array build, so the §5 "fold NULL as 0" bug is still a planted
mutant the suite catches (see §4).

### Assumptions
- SUM/AVG are generated only over numeric inputs (I32/I64/F64); MIN/MAX/COUNT over
  any type. Batches are non-empty (Operator contract) — MIN/MAX kernels require
  `n≥1`, satisfied by the operator.
- Output never routes through `compact_column`, so the tracked 0-row
  `compact_column` abort is **not reachable** from the aggregate path; the
  empty-input cases (global → 1 row, keyed → 0 rows) are produced directly and
  pass under ASan. (Carry-forward note, not fixed here per the brief.)

### ICRs
**None.** No frozen header was edited.

---

## 3. Exact commands

Build (Release-style dev build):
```
cmake -S . -B build
cmake --build build -j4
```

Run the full suite (all 23 tests, incl. WP-5):
```
cd build && ctest --output-on-failure
```

GROUP BY oracle diff — BOTH backends. DuckDB is auto-enabled because the
amalgamation is staged at `third_party/duckdb/`; CMake prints
`DuckDB backend ENABLED` and defines `QE_WITH_DUCKDB` (so `duckdb_available()` is
true and the test diffs every case against DuckDB **and** the reference):
```
cmake -S . -B build && cmake --build build --target aggregate_differential_test -j4
./build/aggregate_differential_test --seed 20260614
# -> 7 cases, 221 assertions, SUCCESS (random sweep + single/composite/global/
#    skew/all-null/empty, each vs reference AND DuckDB)
```

scalar==vector gate (direct kernels + end-to-end global via kVector/kScalar):
```
cmake --build build --target agg_scalar_vector_test -j4
./build/agg_scalar_vector_test --seed 20260614   # 4 cases / 123 assertions
```

Sanitizers (ASan + UBSan) — re-run for WP-5:
```
cmake -S . -B build-asan -DENABLE_SANITIZERS=ON
cmake --build build-asan --target agg_scalar_vector_test aggregate_differential_test agg_mutation_test -j4
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./build-asan/agg_scalar_vector_test     --seed 20260614
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./build-asan/aggregate_differential_test --seed 20260614
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./build-asan/agg_mutation_test           --seed 20260614
# all SUCCESS. (LeakSanitizer is unsupported on macOS/arm64; ASan+UBSan run.)
```

Replay a seeded run (determinism, RIGOR.md rule 5): every binary prints
`[wp1] seed=…` and `--seed N` replays it, e.g.
`./build/aggregate_differential_test --seed 5707195668127091553`.

---

## 4. Mutation self-test (RIGOR.md rule 4)

Command + output showing the differential FLAGS each mutant while the REAL
operator PASSES the identical diff:
```
cmake --build build --target agg_mutation_test -j4
./build/agg_mutation_test --seed 20260614
```
Output (abridged):
```
TEST CASE: MUTATION: the REAL aggregate passes the differential        [pass]
MESSAGE: kFoldNullInSum caught: value mismatch ... engine=0 oracle=NULL
MESSAGE: kEmptyGroupZero caught: null mismatch at canonical row 0 col 3: engine=0 oracle=NULL
MESSAGE: kGroupTailOffByOne caught: row count mismatch: engine=2 oracle=3
[doctest] test cases: 4 | 4 passed | 0 failed
```
Planted mutants (faithful copies of the grouped accumulate/finalize/output loops,
reusing `agg_internal.h`, one bad step each; not linked into any engine target):
- **kFoldNullInSum** — the §5 named bug: SUM/AVG fold a NULL as 0 **and** count
  it → an all-null group emits 0/seen instead of NULL.
- **kEmptyGroupZero** — a zero-non-null group emits a concrete value instead of
  NULL.
- **kGroupTailOffByOne** — output batching drops the last group (short final
  batch → row-count mismatch).

The kernel-level scalar==vector test additionally targets the SIMD-tail
boundary lengths (the off-by-a-lane hotspot) directly.

---

## Message for the final review

WP-5 (hash aggregation / GROUP BY) is complete and green end-to-end, including the
authoritative DuckDB differential (M2). No frozen interface was touched — **no
ICR**. One design decision to ratify: **grouped (multi-key) accumulation is scalar
scatter; only the global reduction is vectorized** (Highway kernel + scalar twin +
gate), mirroring the WP-4 probe-walk precedent — portable SIMD scatter-add with
duplicate group ids isn't safely expressible in Highway. The public
`ops/aggregate.h` surface is unchanged by this, so a future grouped-SIMD
optimization (e.g. conflict-detection) is a drop-in. Carry-forward confirmed: the
aggregate path deliberately avoids `compact_column`, so the tracked 0-row
`compact_column` abort is not reachable here (empty-input cases pass under ASan).
`ops/aggregate.h`'s `AggSpec`/`AggFunc` are designed for WP-8's dataframe builder.
