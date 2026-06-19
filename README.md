# A vectorized, columnar query engine — validated byte-for-byte against DuckDB

**Everything is a typed column batch flowing through composable, pull-based
vectorized operators.** A query is a tree of operators; each pulls ~2048-value
column batches from its children and processes them in tight, SIMD-friendly loops
that keep the working set in cache. **Correctness** is the engine agreeing with
**DuckDB** byte-for-byte on every query, on two ISAs (ARM/NEON today, x86/AVX-512
when the headline box is rented), enforced by a **self-validated differential
oracle**. **Speed** is keeping the batch in cache and letting a portable SIMD layer
emit wide vector code — proven by same-machine *relative* ratios (vectorized-vs-scalar,
engine-vs-DuckDB) and a roofline that labels each operator compute- or bandwidth-bound.

This is **not** a SQL database. It is an execution engine: vectorized, columnar,
SIMD, validated against DuckDB, with tick-data operators (as-of joins, time windows,
Gorilla compression), dictionary-encoded strings, and optional morsel-parallel
execution. It is built **from scratch** — no query-engine or dataframe library
(DuckDB, Arrow-as-a-library, Polars, DataFusion, Velox, pandas) anywhere in the core;
DuckDB appears *only* in the test/oracle targets as the golden model.

> **Honesty up front (read [§ Mac-vs-x86](#the-honest-part-mac-vs-x86) before believing a number).**
> All performance numbers here are captured on an Apple M5 (ARM/NEON) and are
> **preliminary / relative-only**. NEON is 128-bit; AVX-512 is 512-bit — the Mac
> *understates* the vectorization win and cannot give a credible memory-bandwidth
> roofline or core-scaling curve. The headline AVX-512 numbers and the roofline come
> from a future x86 run; the same harness reruns there with **zero code changes**
> (nothing hardcodes ISA, vector width, cache size, or core count).

---

## What's inside

| Layer | Module | What it is |
|---|---|---|
| Columnar format | `core/` | `Buffer` / `Column` / `Batch` / `Schema` / `SelectionVector`, validity bitmap, all-valid fast path, dictionary-encoded strings (`StringDict`) |
| Portable SIMD | `simd/` | [Google Highway](https://github.com/google/highway) wrappers + an independently-written **scalar twin** of every kernel |
| Expressions | `expr/` | typed expression IR + vectorized & scalar kernels (arith / compare / logical / cast) with documented null propagation |
| Operators | `ops/` | pull-based `scan` · `filter` · `project` · `hashtable` · `aggregate` · `join` · `sort` (numeric **and** VARCHAR keys) |
| Plan | `plan/` | physical plan IR + a fluent **dataframe-style builder** (no SQL parser, by design) |
| Tick-data (Phase 2) | `tsx/` | `asof` (as-of join) · `window` (tumbling + sliding) · `compress` (Gorilla / delta-of-delta) |
| Parallel exec | `exec/` | optional **morsel-driven** parallel driver wrapping the frozen operators (TSan-clean; off by default) |
| Oracle | `oracle/` | the DuckDB differential runner, the seeded generators, and the **mutation catalog** |
| Bench | `bench/` | percentile + roofline harness, validity gate, coordinated-omission correction, host capture |

Every operator is a `pull`-based node (`Operator::next() -> optional<Batch>`), so each
is unit-testable in isolation and composes into arbitrary trees. The dataframe builder
*is* the query surface:

```cpp
auto plan = scan(trades)
    .filter(gt(col(Type::F64, /*price*/2), lit(Scalar::f64(100.0))))
    .join(scan(quotes), /*left*/{0}, /*right*/{"sym"}, JoinType::Inner)
    .aggregate({"sym"}, {AggSpec::count_star("n"), AggSpec::avg(2, "avg_px")})
    .sort({SortBy{"n", SortDir::Desc}});
auto op = plan.build();          // -> std::unique_ptr<Operator>
```

The **same plan** lowers to the engine operator tree *and* renders to the SQL the
DuckDB oracle runs — one description drives both backends, which is what makes the
differential honest.

---

## Correctness is the headline: a self-validated oracle

Every operator is diffed against DuckDB on the same inputs, over seeded random
schemas, data, and queries. Float aggregates compare with a relative+absolute epsilon
(summation order differs); everything else is exact. Unordered results are
canonicalized on all output columns before diffing; `ORDER BY` is compared
positionally.

**But a checker that cannot fail proves nothing.** The oracle's own credibility comes
from a **mutation catalog** — a registry of deliberately-broken engine variants that
the differential suite *must* flag while the real engine passes. The meta-test asserts
≥6 entries, that every §12 hazard class is spanned, and that for **every** entry the
clean engine passes **and** the mutant is caught:

```
catalog entries: 17 (floor 6)   —   every entry: clean_passes=true, mutant_flagged=true
```

Hazards covered include SIMD tail/remainder, Kleene null propagation, the all-valid
fast path skipping a real null, hash-probe overflow, aggregation overflow,
selection-vector aliasing, float→int rounding, and operator/plan orchestration
(join/sort/agg/lowering, as-of boundary, **window frame**, **compression decode**,
**string-key-by-value**, **parallel partial-aggregate merge**).

Two lessons baked into the process:
- **DuckDB must actually run.** The first "green" CI was stale reference-only
  binaries where DuckDB never linked. Acceptance now verifies real DuckDB symbols
  (`nm`) and that a planted bug is caught *through the DuckDB path* with the
  independent reference bypassed.
- **The reference must be independent.** Each reference oracle recomputes results with
  plain std-library logic sharing no code with the engine path, so "engine == reference"
  is meaningful alongside the authoritative DuckDB diff.

```bash
# the consolidated mutation catalog — clean passes, every mutant is flagged
ctest --test-dir build-asan -R mutation_catalog_meta --output-on-failure
```

---

## The pinned bug (a real one, with a regression)

**`F64 -> I64` cast: vectorized rounding diverged from scalar on large integral doubles.**

The vectorized cast rounded with `Trunc(x + copysign(0.5, x))`. For `|x| >= 2^52` a
double has no fractional bits, so adding `0.5` rounds *up* to the next representable
integer — corrupting values that were already exact integers. The scalar twin used
`std::round` and disagreed. It was **data-dependent** (only large magnitudes) and only
visible because the scalar twin is independent code, not the vector path with SIMD
switched off.

- **Found by:** the `scalar == vector` property test, seeded.
- **Fix:** `expr/cast_kernels.cpp::RoundAway` — truncate, then add an *exact* fractional
  bump, matching `std::round`.
- **Regression:** `expr_property_pinned_castbug --seed 14745959599513406255` (and a
  value-level case in `expr_scalar_vector_test`), both confirmed to fail before / pass
  after by reverting the fix.

A second, smaller one is pinned too: `compact_column(n==0)` aborted on empty input (a
real 0-row crash); fixed signature-preserving, with `compact_empty_test` that SIGABRTs
without the fix and a 0-row differential vs DuckDB.

---

## The tick-data angle (Phase 2)

Three operators aimed at quant/time-series workloads, each diffed against DuckDB:

- **As-of join** (`tsx/asof.h`) — backward nearest-preceding-timestamp join per
  partition key, INNER/LEFT, optional tolerance window. Validated against DuckDB's
  `ASOF JOIN` on irregular ticks with gaps, exact-timestamp boundary ties, composite
  keys, and NULL semantics. The classic `>`-vs-`>=` boundary bug is a mutation self-test.
- **Time windows** (`tsx/window.h`) — **tumbling** (time-bucketed `GROUP BY`,
  `bucket = (t/W)*W`) and **sliding** (running aggregate over `ROWS BETWEEN P PRECEDING
  AND CURRENT ROW`), reusing the Phase-1 aggregate vocabulary. Validated against
  DuckDB's `time_bucket`-style `GROUP BY` and window functions.
- **Time-series compression** (`tsx/compress.h`) — from-scratch **Gorilla XOR** (F64),
  **delta-of-delta + zig-zag + varint** (timestamps / int64), and a **compressed scan**
  that decodes batch-by-batch so the decoded stream feeds the engine end-to-end. The
  round-trip is lossless and bit-exact across the whole double domain (NaN payloads,
  ±0.0, ±inf, subnormals) and preserves the validity bitmap exactly.

Compression on a synthetic tick stream (`host=mac-m5`, `isa=neon`, **preliminary**):

| column | codec | ratio |
|---|---|---|
| timestamp | delta-of-delta | **2.70×** |
| price (F64) | Gorilla XOR | **1.47×** |
| volume (I64) | delta-of-delta | **2.74×** |
| **overall** | | **2.12×** |

```bash
cmake --preset release && cmake --build build --target bench_compress -j8
./build/bench_compress --seed 20260614 --rows 2000000 --reps 11   # emits the JSON above
```

---

## Two more, both validated against DuckDB

- **Dictionary-encoded strings (`VARCHAR`).** A `STR` column stores int32 dictionary
  codes plus an out-of-band `StringDict` (code → bytes). The subtle part — and the
  planted mutation that proves the test — is that group-by/join keys are compared **by
  value, not by code**: the two sides of a join carry independent dictionaries, so the
  same string gets different codes and must be canonicalized into a shared value-id
  space first. Validated against DuckDB `VARCHAR` group-by / join / order-by / MIN-MAX.
- **Morsel-driven parallel execution (optional).** An additive layer (`exec/`) that
  wraps the frozen pull-based operators: the driving scan is split into morsels pulled
  from a shared atomic cursor, each worker runs a private operator pipeline, and a final
  exchange concatenates (streaming/join) or merges per-key partial aggregates. It
  changes **no** frozen interface, is **TSan-clean**, and the oracle proves it produces
  results *identical to single-thread* across thread counts {1,2,4,8} and morsel sizes.
  (Speedup/scaling is an x86 deliverable — see below; the Mac result is correctness.)

---

## Performance — preliminary / relative-only (host=mac-m5, isa=neon)

> Apple M5 (4 P-cores + 6 E-cores), `hwy=NEON`, AppleClang `-O2 -march=native`.
> macOS has no `performance` governor, no reliable core pinning, and heterogeneous
> P/E cores, so **absolute** latency and **core-scaling** are not reported here. These
> are *relative* ratios on one machine, where host variance cancels.

**Vectorized-vs-scalar, per operator** (speedup = scalar/vector at p50):

| operator | rows | vec p50 (µs) | scalar p50 (µs) | speedup | GB/s | roofline |
|---|---|---|---|---|---|---|
| aggregate | 262144 | 1097.7 | 827.4 | 0.75 | 1.91 | bandwidth-bound |
| expr | 262144 | 249.9 | 219.1 | 0.88 | 33.57 | bandwidth-bound |
| hashtable | 262144 | 2048.0 | 1851.4 | 0.90 | 1.54 | bandwidth-bound |
| sort | 262144 | 25952.3 | 26476.5 | 1.02 (p99 1.36) | 0.40 | bandwidth-bound |
| permutation gather | 1048576 | 622.6 | 704.5 | **1.13** | — | — |

**This table is an honest negative result, not a missing win.** On Mac/NEON these
columnar ops are **bandwidth-bound**, and the "scalar" twin is *compiler
auto-vectorized*, so vector ≈ scalar — exactly what §2 predicts (NEON is narrow; the
twin isn't truly scalar). The permutation **gather** (1.13×) is the cleanest separable
win. The real vectorization story is an **x86/AVX-512** result with a true scalar
baseline (or a deliberately compute-bound kernel) — that is the headline pending the
rented box.

**Engine-vs-DuckDB:** *deliberately not quoted as a number here.* The current harness
reloads the table into DuckDB on every call, so the wall-clock ratio is dominated by
DuckDB's ingest, not query execution — it would be a misleading "speedup." A
load-once / time-the-run-only seam lands with the x86 run; only then is an
engine-vs-DuckDB ratio honest enough to print. (DuckDB is a mature, heavily optimized
engine; a fair single-query comparison is not assumed to favor this engine — the point
of the differential is *correctness*, not beating DuckDB.)

**Coordinated-omission correction self-test.** Under sustained open-loop load,
latency is measured from *intended* send time; an injected 500 ms stall must surface on
the intended-time tail while the naive actual-time tail hides it:

| run | thr/s | intended p99.9 | actual p99 | CO factor |
|---|---|---|---|---|
| steady | 2000 | 5.6 ms | 0.15 ms | 5.6× |
| +500ms stall injected | 2000 | **503 ms** | 0.15 ms | **3099.7×** |

The validity gate independently **rejects** any run contaminated by background load or
thermal throttling rather than reporting it.

```bash
# regenerate every number above from the engine, then the tables/plots FROM saved JSON
scripts/run_bench.sh                       # runs the drivers -> bench/results/*.json
python3 scripts/plot_results.py            # plots + summary regenerate WITHOUT re-running the engine
cat bench/results/summary.txt
```

---

## The honest part: Mac-vs-x86

We develop on Apple Silicon and will benchmark headline numbers on a rented x86
bare-metal box (Sapphire Rapids class, AVX-512). This split is fine **because the SIMD
layer is portable from day one and the harness is host-parameterized** — the x86 rerun
is launch-and-go.

What the Mac numbers here **can** claim: all logic/operator/oracle correctness
(ISA-independent), cross-ISA coverage, and a *directional* relative story
("vectorization helps, roughly this order of magnitude"). What they **cannot** claim,
and this README does not: any AVX-512-specific magnitude, a credible memory-bandwidth
roofline, clean core-scaling, or stable absolute latency. Every result is tagged
`host=` + `isa=` and labeled preliminary; the roofline ridge on Mac is *assumed*, not
measured.

**Pending the x86 run (M5):** AVX-512 vec-vs-scalar with a true scalar baseline, the
credible roofline, a load-once engine-vs-DuckDB number, and (if built) a
core-scaling curve. The drivers rerun unchanged.

---

## Build & reproduce

Requires CMake ≥3.20 and a C++20 compiler. Google Highway is vendored. DuckDB is
**optional** — stage the v1.1.3 amalgamation at `third_party/duckdb/duckdb.cpp` to
enable the authoritative differential (absent it, the independent reference oracle runs
and CI stays green).

```bash
# sanitizer-gated build (ASan + UBSan) — the correctness gate
cmake --preset asan && cmake --build build-asan -j8
ctest --test-dir build-asan --output-on-failure          # full suite, incl. DuckDB diffs when staged

# from-scratch enforcement (no forbidden libraries in core/simd/expr/ops/plan/tsx)
scripts/check_forbidden_includes.sh

# release build for the bench drivers
cmake --preset release && cmake --build build -j8
scripts/run_bench.sh
```

All randomized tests print their seed and replay from `--seed N`. Sanitizers (ASan +
UBSan) are a build gate; the parallel layer adds a **TSan** gate. The full integrated
suite is **51/51 green** under the `asan` preset with DuckDB staged.

---

## Non-goals (deliberate scope cuts — this is about execution *technique*)

No SQL parser (the dataframe/plan API is the surface, by design) · no
persistence/WAL/storage engine (in-memory) · no transactions/MVCC · no distributed
execution · no cost-based optimizer (physical plans are builder-assembled). (Dictionary
strings and multi-threaded execution *were* deferred scope cuts — both are now built,
optional, and oracle-validated.)

---

## Layout

```
core/ simd/ expr/ ops/ plan/        from-scratch engine (no query-engine libs)
tsx/                                Phase-2 tick-data operators (asof, window, compress)
exec/                               optional morsel-driven parallel execution (TSan-clean)
oracle/                             DuckDB differential + seeded generators + mutation catalog
bench/ scripts/                     percentile/roofline harness, host capture, plot-from-JSON
tests/                              unit + property + differential + regression + mutation self-tests
```
