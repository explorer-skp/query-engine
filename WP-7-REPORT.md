# WP-7 — Sort (ORDER BY)

Pipeline-breaking sort operator with a comparison path and an integer/timestamp
radix fast-path, multi-column ASC/DESC + NULLS FIRST/LAST, whole-row gather, and a
DuckDB-authoritative ORDER BY differential under **positional** compare. Carries
the engine's first vec-vs-scalar number (the permutation gather).

Status: **green** — radix == comparison, gather scalar == vector, ORDER BY diffs
green vs DuckDB + reference on seeded random data + edges, 5 mutants caught under
positional compare, ASan/UBSan clean, from-scratch gate clean.

---

## 1. `ops/sort.h` — the public surface (WP-8-consumable)

```cpp
enum class SortDir   { Asc, Desc };
enum class NullOrder { First, Last };
struct SortKey { std::uint32_t col; SortDir dir = Asc; NullOrder nulls = Last; };

class Sort : public Operator {
  Sort(std::unique_ptr<Operator> child, std::vector<SortKey> keys);
  enum class Path       { kAuto, kComparison, kRadix };   // test seam
  enum class GatherPath { kVector, kScalar };             // test seam
  void set_path(Path);          // before open(); WP-8 ignores (defaults correct)
  void set_gather_path(GatherPath);
  // + the four frozen Operator methods
};
```

`SortKey.col` is the **child-output column index** (= the query output column).
`output_schema()` returns the child's schema unchanged — sort reorders rows, never
columns/types. The two enums are the entire ORDER BY vocabulary; the `Path` /
`GatherPath` seams exist only so the two correctness cross-checks below can drive
each path on identical inputs (production uses `kAuto` / `kVector`).

No frozen header was edited. `Sort` conforms to the frozen `ops/operator.h`
contract: `open()` drains + sorts, `next()` emits sorted rows in `<=2048`-row
dense batches, `close()` releases state.

## 2. Multi-column strategy

Left-to-right precedence: `keys[0]` is primary. Both paths are **stable** on the
child's row order (rows with an equal key tuple keep their original relative
order), so the two paths produce the **same permutation** on any input.

* **Comparison path** (`build_perm_comparison`): `std::stable_sort` of a row-index
  array under a lexicographic tuple comparator (`less_row`) that reads raw column
  values and branches per key on null / direction. Handles any type / mix.
* **Radix fast-path** (`build_perm_radix`, eligible when **every** key is
  I32/I64/TS): each row's key tuple is normalized into one fixed-width,
  **memcmp-able big-endian byte string** — `[1 null-indicator byte][value bytes]`
  per key, key0 most significant — then a **stable LSD byte radix** (least-
  significant byte first) sorts the permutation. Encoding details:
  * order-preserving value: signed → unsigned **sign-bit flip** (`v ^ 0x80…`),
    emitted big-endian, so bytewise ascending == numeric ascending;
  * **DESC**: bytewise complement of the value bytes (reverses order);
  * **NULLS FIRST/LAST**: the null-indicator byte (0 vs 1) — set by null order
    **only**, never by ASC/DESC, because SQL null placement is absolute. Null rows
    get constant 0 value bytes, so they stay grouped and are placed solely by the
    indicator byte.
  Multi-pass over all `W` bytes gives full multi-column order; LSD stability
  carries lower-significance bytes through.

The two are **independent algorithms** (not a templated body), per §3/D17, so
`radix == comparison` is a meaningful check: a sign-bit / null-byte slip surfaces
as a disagreement (and is exactly the planted `kRadixSignBug` mutant).

Whole rows move via `sort_internal::gather_rows`: it gathers each output column's
data bytes by the permutation through the **Highway gather kernels** (`gather32`/
`gather64` + scalar twins, `simd/gather_kernels.h`) — out-of-place, so the §12
aliasing hazard cannot arise — and rebuilds validity with a scalar loop (the
bitmap is not the SIMD story). BOOL (1 byte) uses `gather8_scalar` on both paths
(no vector twin exists; documented).

### Why no new Highway kernel (and where the SIMD win is)

The radix histogram/scatter is inherently sequential control flow (duplicate
buckets can't be scatter-added portably with Highway) — the same call the codebase
already makes for the WP-4 probe walk and the WP-5 grouped scatter; its
correctness is pinned by `radix == comparison` + the differential + the mutant,
not by `scalar == vector`. The genuine data-parallel step in a sort pipeline is the
**whole-row permutation gather**, which the brief explicitly offers as the
vec-vs-scalar lever ("or vec-gather vs scalar-gather"). So WP-7 **reuses** the
existing WP-1 gather twins rather than hand-rolling a new intrinsic kernel, and the
headline ratio is gather-vec vs gather-scalar (§6).

## 3. Ordered-mode comparator — coexisting with the unordered one

`compare_result_sets(engine, oracle, bool ordered = false)` gained one defaulted
parameter, so every existing call site (WP-3..WP-6) keeps the historical UNORDERED
behavior untouched:

* `ordered == false` (default): D12 canonicalization — sort both sides on all
  output columns, then diff. Unchanged.
* `ordered == true`: **no canonicalization**; rows compared **positionally** in
  emitted order. This is what makes a wrong sort observable — a canonicalizing
  compare would re-sort both sides and hide it.

Cell-level rules are identical in both modes: exact integers, exact nullness, and
the **D11 F64 epsilon still applies per cell** (positional compare does not make
floats exact). `run_differential` selects the mode automatically with
`q.has_order_by()`, so ORDER BY queries compare positionally and everything else
stays unordered. The join differential is untouched (joins are unordered).

## 4. Float / NaN / -0.0 determinism vs DuckDB

Both engine paths and DuckDB order F64 by IEEE `<`. The hazard for **positional**
compare is an ambiguous row order under a correct sort:

* **Ties.** The ORDER BY generator (`gen_order_by_query`) projects every column and
  appends all not-yet-used columns as deterministic tiebreakers (ASC NULLS LAST),
  making the sort a **total order**. Any remaining tie is therefore between rows
  that are *element-wise equal in every column*, whose relative order is
  unobservable — positional compare passes regardless of how engine vs DuckDB
  broke it. (Genuine partial-key / stability behavior is checked separately
  against the **reference** oracle, which is stable like the engine; see the
  `kUnstableTiebreak` mutant.)
* **NaN / -0.0.** The data generators emit only finite, non-NaN floats and do not
  produce a distinct `-0.0` (uniform real draw), so F64 ordering is fully
  determined and matches DuckDB. The radix path never touches F64 (it is integer-
  only by eligibility), so the NaN/-0.0 encoding question does not arise there. If
  NaN/-0.0 generation is ever enabled, the total-order tiebreak above keeps the
  positional compare honest only when the tuple stays unique — documented here as
  the constraint.

The reference oracle sorts its own `ResultSet` rows with an **independent**
`std::stable_sort` over materialized `Cell`s (shares no code with the engine's
radix/comparison), so `engine == reference` under ordered compare is a real check.

## 5. Mutation self-test (positional compare — verified to bite)

`ops/sort_mutants.{h,cpp}` — a faithful copy of the real drain/permutation/emit
reusing `sort_internal` (materialize + gather), one planted defect each. All five
are caught under positional compare; the real operator passes the same diff:

| Mutation | Defect | Caught by |
|---|---|---|
| `kDescSortsAsc` | DESC sorted ASC | ref + DuckDB |
| `kNullsFlipped` | NULLS FIRST/LAST flipped | ref + DuckDB |
| `kRadixSignBug` | radix omits the sign-bit flip (negatives sort last) | ref + DuckDB |
| `kUnstableTiebreak` | equal-key rows emitted in reversed order | ref (DuckDB tie order undefined) |
| `kEmitTailOffByOne` | drops the last sorted row | ref + DuckDB |

`kUnstableTiebreak` is meaningful only against a **stable** oracle on a
**partial** key (distinct payload), so it is checked vs the reference; the others
are total-order cases checked vs both. Each was confirmed to flip to *failing*
under the canonicalizing (unordered) comparator → *passing* under positional, i.e.
the ordered-mode comparator is what makes the sort mutants observable.

## 6. Measurement — permutation gather, vec vs scalar (PRELIMINARY / RELATIVE-ONLY)

`host=mac-m5 isa=neon` · Apple M5 · AppleClang 21 · `-O2 -march=native` ·
Highway target NEON. **Mac → relative-only** (RIGOR.md §2): no absolute /
AVX-512 / roofline / bandwidth claim; the JSON carries `"preliminary": true`.

`gather64` over 1,048,576 × 8-byte elements with a scattered index stream, 200
timed iterations:

| | p50 | p99 |
|---|---|---|
| vector (`gather64_vec`) | 614,399 ns | 946,042 ns |
| scalar (`gather64_scalar`) | 696,319 ns | 1,072,416 ns |
| **speedup (scalar/vec)** | **1.13×** | **1.13×** |

A scattered 64-bit gather is memory-latency bound, so the SIMD win is modest and
honest — the bottleneck is the random loads, not lane width. JSON saved at
`bench/results/wp7_sort_gather_mac-m5_neon.json`. The `scalar == vector` gate
(boundary lengths + end-to-end) proves the two paths agree on values; this is the
speed delta. (`gather32` is also gate-checked for equality but not separately
timed.)

## 7. Reproduce commands

```bash
# Build (Release; DuckDB auto-enabled because the amalgamation is staged)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# scalar==vector (gather) AND radix==comparison
./build/sort_scalar_vector_test --seed 20260614

# ORDER BY differential vs the independent reference AND DuckDB (positional, D12).
# DuckDB is ON whenever third_party/duckdb/duckdb.cpp is staged (QE_WITH_DUCKDB);
# this build links it — see the "DuckDB backend ENABLED" line at configure time.
./build/sort_differential_test --seed 20260614

# Mutation self-test: the 5 mutants are CAUGHT (positional), real operator PASSES
./build/sort_mutation_test --seed 20260614

# Replay any failure from its printed seed, e.g.
./build/sort_differential_test --seed <N>

# Sanitizers (ASan + UBSan) — the gate
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-asan -j8 --target \
  sort_scalar_vector_test sort_mutation_test sort_differential_test
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ./build-asan/sort_differential_test --seed 20260614   # (and the other two)

# vec-vs-scalar number (preliminary / relative-only; tagged host+isa)
./build/bench_sort --n 1048576 --iters 200 --seed 20260615 \
  --out bench/results/wp7_sort_gather_mac-m5_neon.json

# From-scratch gate (no DuckDB/SQLite/Arrow/... in core/simd/expr/ops/plan/tsx)
bash scripts/check_forbidden_includes.sh
```

All sanitizer runs re-run and reported green; `sort_*` pass for seeds
`{20260614, 111, 999, 42424242, 7, 20260615}` vs DuckDB.

## 8. Assumptions

* ORDER BY keys reference **output** columns (post project/aggregate). SQL renders
  them by **1-based ordinal** (`ORDER BY 1 DESC NULLS LAST, …`), which binds to the
  SELECT position = `query_output_schema` order = engine output order, so no name
  resolution is needed and NULLS placement is always explicit (never DuckDB's
  default).
* Radix eligibility is "all keys I32/I64/TS". F64/BOOL keys route to the
  comparison path (BOOL is trivially radix-able but kept out per the D10 scope —
  comparison handles it).
* The known reviewer-owned `compact_column` n==0 abort is **not reachable**:
  the drain materializes columns directly (never via `compact_column`), and empty
  input emits zero rows.

## 9. ICRs

**None.** No frozen interface was edited. `compare_result_sets` grew a *defaulted*
parameter (additive, backward-compatible) and `LogicalQuery` an *optional*
`order_by` field — both in the non-frozen `oracle/` layer.

## 10. Note on the concurrent WP-6 (join) worker

WP-6 and WP-7 share the same working tree and both extend `oracle/`. I confined my
edits to ORDER BY plumbing and added the ordered comparator mode without touching
join code: `compare_result_sets` got a defaulted `ordered` arg (join keeps the
default), `LogicalQuery` got `order_by`, `build_engine_pipeline` appends `Sort`,
`select_sql` appends `order_by_clause`, the reference oracle gained an independent
`apply_order_by`, and `gen_order_by_query` was added. All existing oracle/agg/join
call sites compile and the WP-3/WP-5 differential + mutation tests still pass
unchanged (verified). No join file was modified.
