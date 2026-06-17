# WP-10b Report — Parallel execution (morsel-driven)

**Scope:** an *optional, additive* parallel-execution layer (decision D15) that runs a
built `plan::Plan` across worker threads and returns the **same logical result** as the
single-thread plan path, proven by the oracle. **No frozen-interface change.** New gate:
**TSan green.**

## What landed

New self-contained layer in **`exec/`** (new dir + `qe::exec` namespace), composing the
frozen operators — nothing in `core/ ops/ expr/ plan/ tsx/ simd/` changed:

- **`exec/morsel_scan.{h,cpp}`** — `MorselScan`: a frozen-`Operator` leaf that scans a
  **64-aligned** contiguous row range `[start,end)` of a `Table`, zero-copy, identical
  per-batch view logic to `ops/scan.cpp` (only the row window + the `start % 64 == 0`
  precondition differ). 64-alignment keeps the validity-word sub-view exact (the same
  invariant `Scan` documents).
- **`exec/parallel.{h,cpp}`** — `ParallelEngine` / `run_plan_parallel(plan, cfg)`.
  `ParallelConfig{threads, morsel_rows}`: **thread count is configuration**
  (`0 ⇒ std::thread::hardware_concurrency()`), never a baked constant; `morsel_rows`
  is a multiple of 64. Returns a materialized `oracle::ResultSet`.
- **`exec/parallel_mutants.{h,cpp}`** — `qe::mutant::ParallelMutation::kMergeDropPartial`
  (the planted parallelism bug; separate lib `qe_parallel_mutants`).

### Morsel model (Leis et al., scoped to what the oracle can verify)

- **Morsel** = a 64-aligned row range of the **driving scan** (the bottom-left base table
  of the pipeline). Worker threads pull morsels from a **shared `std::atomic` cursor**
  (a shared dispenser; work-stealing not required). One morsel = one full **private**
  sub-tree execution over that row range.
- **Per-pipeline instances:** each morsel runs through a PRIVATE operator tree
  (`MorselScan` leaf + frozen operators rebuilt via the engine's own lowering). No engine
  state is shared between threads.
- **Exchange / merge, by the plan's top operator:**
  - **Scan→Filter→Project (streaming):** embarrassingly parallel — each morsel emits a
    partial result; the exchange **concatenates** (order-free; the diff canonicalizes,
    D12).
  - **Join:** the **probe** side is morsel-partitioned; each worker builds its **own**
    build-side hash table (single-thread build *per worker*) and probes its probe-morsels;
    outputs concatenate. (The brief permits per-worker build; the shared-build-once
    optimization is future work — see *Assumptions*.)
  - **Aggregate (group-by):** each **worker** accumulates a **private** partial hash
    aggregate over all its morsels (running a frozen `Aggregate` per morsel, then folding
    its rows with plain combine). **AVG is decomposed to SUM+COUNT** so partials are
    mergeable. The **cross-worker merge** combines per group key — sum the sums/counts,
    min the mins, max the maxes; AVG = merged-sum / merged-count. Global (zero-key)
    aggregate merges to exactly one row (including over empty input).
  - **Sort (ORDER BY):** the upstream pipeline feeding the Sort is parallelized; the
    **final Sort runs single-thread** over the merged input. This is the brief's
    explicitly-sanctioned simpler choice (parallelize upstream, sort single-thread) — it
    keeps the positional diff honest and avoids a bespoke k-way merge. **Documented as a
    deliberate scope choice.**
- **Out of scope (run single-threaded, and we say so):** `AsofJoin`/`Window`/
  `CompressedScan`, or any pipeline-breaker on the driving path. `ParallelEngine::supported()`
  returns false for those and `run()` falls back to a **correct single-thread**
  `plan.lower()` execution.

### How shared state is made race-free (the TSan argument)

The **only** shared-mutable object during compute is the atomic morsel cursor
(`fetch_add`). Everything else is **worker-private**: each worker builds its own operator
trees, its own `OwnedBatch` storage, and (for group-by) its own partial-aggregate map
indexed by its own worker id. All combining happens **after `join()`** — a
happens-before barrier — on the main thread. The `Table`, the `Plan` node tree, and the
`expr::Expr`/`AggSpec` vectors are read-only and shared by const reference (and
`expr::Expr`/`Plan` are `shared_ptr` value handles, so copies into per-morsel operators
are refcount bumps, not data races). Highway's runtime dispatch is warmed single-threaded
(the driver runs the `threads=1` case — main thread only — before any multithreaded run,
and the tests compute the single-thread baseline first).

### The identical-to-single-thread argument

For **streaming/join**, every base row lands in exactly one morsel and each probe row is
matched against the full build table, so the concatenation of per-morsel outputs is the
exact set the single-thread tree produces (canonicalized diff). For **group-by**, partial
aggregation + an exact additive/min/max merge is algebraically the single-thread group
state (AVG via SUM+COUNT; null/empty semantics preserved because the partials come from
the frozen `Aggregate`, and a group is NULL iff no morsel saw a non-null). For **sort**,
the input set is identical and the single final sort defines the order. The differential
asserts this directly: **parallel == single-thread == reference == DuckDB** at threads
{1,2,4,8} × morsel sizes {64,128,512}.

## Mutation self-test + catalog (rule 4)

Planted **`kMergeDropPartial`**: the cross-worker partial-aggregate merge **overwrites
instead of accumulates** — for a group key present in ≥2 worker partials it keeps only the
last and **drops the rest** (the brief's canonical "partial-aggregate merge that drops one
partial"). It is a bug that **bites ONLY with >1 worker**: the merge hooks fire solely
when a group is split across worker partials, so at 1 thread the mutant is byte-identical
to the clean driver. Separate code: `qe::mutant` lib derives `ParallelEngine` and overrides
only the three protected merge hooks.

- `tests/wp10b_parallel_mutation_test.cpp` proves: real driver clean at 1/2/4/8; mutant
  **clean at 1 thread**, **flagged at 2/4/8** vs single-thread AND reference AND DuckDB.
- Catalog entry `parallel_merge_drop_partial` (hazard `kOperatorOrchestration` — **no new
  `Hazard` enumerator**) wired into `mutation_catalog.cpp` + `tests/catalog_checks_parallel.cpp`
  + the `mutation_catalog_meta_test` target. Catalog now **16 entries**, all flagged.

## Build / CMake

- New libs: `qe_parallel` (links `qe_oracle` for the `ResultSet` currency + `drain_operator`
  + `Threads::Threads`; **never** DuckDB), `qe_parallel_mutants`.
- New tests `wp10b_parallel_differential_test`, `wp10b_parallel_mutation_test` added to the
  default test set; since the suite has **no per-test preset gating**, they also build/run
  under the **`tsan`** preset automatically (the new gate).
- `scripts/check_forbidden_includes.sh`: added **`exec`** to `MODULE_DIRS` so the new layer
  is covered by the from-scratch gate. *(Minor WP-0-infra touch — flagged for ratification.)*

## Assumptions / ICRs

- **No ICR.** No frozen signature changed (`git diff` clean on all public headers).
- **Per-worker build-side hash table** for joins (not build-once-shared): correct and
  TSan-trivially-clean (fully private), brief-permitted; build-once is a future
  optimization.
- **Top Sort runs single-threaded** over the merged input (brief-sanctioned); only the
  pipeline feeding it is parallelized.
- F64 group keys are keyed by bit pattern — exact for the generators' data (no −0.0/NaN,
  per `generators.h`), the only divergence-free regime.

## Scaling measurement (Mac = NOT credible; deferred)

Per §2 and the brief, **no scaling curve is claimed on this Mac** (no governor, no pinning,
P/E heterogeneity). The deliverable here is **correctness + TSan-green**. The optional
thread-sweep bench driver is **deferred** (it would be Mac-harness-proof only); the credible
curve is a future x86 rerun of unchanged code.

## Commands (reproduce)

```bash
# configure + build (ASan/UBSan AND TSan)
cmake --preset release && cmake --build build
cmake --preset asan    && cmake --build build-asan
cmake --preset tsan    && cmake --build build-tsan

# from-scratch gate (now covers exec/), and prove it bites
bash scripts/check_forbidden_includes.sh
bash scripts/check_forbidden_includes.sh --self-test

# parallel differential at {1,2,4,8} threads × {64,128,512} morsels
#   == single-thread == reference == DuckDB (when staged)
ctest --test-dir build -R wp10b_parallel_differential_test --output-on-failure

# mutation self-test (clean@1, flagged@>1) + catalog meta-test
ctest --test-dir build -R "wp10b_parallel_mutation_test|mutation_catalog_meta_test" --output-on-failure

# sanitizer gates
ctest --test-dir build-asan -R "wp10b_parallel_differential_test|wp10b_parallel_mutation_test|mutation_catalog_meta_test" --output-on-failure
ctest --test-dir build-tsan -R "wp10b_parallel_differential_test|wp10b_parallel_mutation_test" --output-on-failure

# replay a specific seed
./build/wp10b_parallel_differential_test --seed N
```

## Results on this machine (host=mac, isa=neon — correctness only, NOT a speed claim)

- Release full suite: **49/49** (was 47 + the 2 WP-10b tests); no regression.
- DuckDB genuinely linked into the parallel differential (`nm` = 55,868 duckdb symbols)
  and ran; the mutant is caught **via DuckDB** (reference-isolated) at >1 thread.
- ASan/UBSan: **green** on all three WP-10b targets.
- Forbidden-include gate: clean over `core simd expr ops plan tsx exec`; self-test bites.
