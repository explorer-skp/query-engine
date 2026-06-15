# WP-9 Report — Oracle & fuzz framework hardening

**Scope delivered:** the consolidated **mutation catalog** + a **meta-test** that
proves it; the **float→int +0.5** carry-forward wired as a must-flag mutant; the
authorized **`compact_column(n==0)`** frozen-core fix + before/after regression;
**0-row differential** cases against DuckDB; seeded-generator **replay**; cross-ISA
cleanliness. No new vectorized kernel was added, so the scalar-twin rule is **N/A**
(stated explicitly). No forbidden includes added (DuckDB/SQLite stay under
`oracle/` + `third_party/`; my new code is all under `tests/`).

---

## 1. Mutation catalog

A single documented registry now names every must-flag mutant, the §12 hazard it
represents, and the test that catches it. Full inventory: **`MUTATION_CATALOG.md`**.
Executable form: `tests/mutation_catalog.{h,cpp}` (registry) + the conflict-free
check TUs `tests/catalog_checks_*.cpp` + the meta-test
`tests/mutation_catalog_meta_test.cpp`.

**12 registry entries**, spanning every named §12 hotspot + both carry-forwards:

| id | §12 hazard | mutant | catching test |
|----|-----------|--------|---------------|
| simd_tail_remainder | SIMD tail/remainder | `simd::mutant::all_valid_vec_tailbug` | validity_mutation_test |
| null_propagation_kleene | null propagation (Kleene) | `expr::mutant::logic_and_twovalued` | expr_mutation_test |
| all_valid_fastpath_skips_null | all-valid fast path skips a null | `expr::mutant::propagate_nulls_and_allvalidbug` | expr_mutation_test |
| hashjoin_probe_overflow | hash probe overflow / collision | `mutant::HashTable{kFindStopEarly}` | hashtable_mutation_test |
| hash_growth_rehash_drop | hash growth rehash | `mutant::HashTable{kSkipRehashOne}` | hashtable_mutation_test |
| **aggregation_overflow_i32** | **aggregation accumulator overflow** | `catalog::mutant::sum_i64_wrapping_i32` | mutation_catalog_meta_test |
| **selection_vector_aliasing** | **selection-vector index aliasing** | `catalog::mutant::gather_in_place_aliased` | mutation_catalog_meta_test |
| **float_to_int_round_plus_half** | **float→int round via +0.5 (WP-2 carry-fwd)** | `catalog::mutant::cast_f64_to_i64_plus_half` | mutation_catalog_meta_test |
| join_drop_probe_match | join: dropped match | `mutant::HashJoin{kDropProbeMatch}` | join_mutation_test |
| sort_desc_as_asc | sort: DESC as ASC (positional) | `mutant::Sort{kDescSortsAsc}` | sort_mutation_test |
| agg_fold_null_in_sum | agg: NULL folded as 0 in SUM | `mutant::Aggregate{kFoldNullInSum}` | agg_mutation_test |
| plan_drop_sort | plan-lowering: dropped Sort (positional) | `plan::lower_mutant{kDropSort}` | plan_mutation_test |

Bold = the three new WP-9 mutants. The meta-test asserts **(a)** ≥6 entries
(`kMinCatalogEntries`), **(b)** every §12 hazard class is spanned, **(c)** for every
entry the un-mutated path passes **and** the mutant is flagged — against **DuckDB**
when `QE_WITH_DUCKDB` is set (228k DuckDB symbols linked; verified).

**Meta-test output (release, `--seed 20260614`)** — every entry
`clean_passes=true mutant_flagged=true`:

```
[simd_tail_remainder]            mutant=simd::mutant::all_valid_vec_tailbug          clean=true flagged=true
[null_propagation_kleene]        mutant=expr::mutant::logic_and_twovalued            clean=true flagged=true
[all_valid_fastpath_skips_null]  mutant=expr::mutant::propagate_nulls_and_allvalidbug clean=true flagged=true
[hashjoin_probe_overflow]        mutant=mutant::HashTable{kFindStopEarly}            clean=true flagged=true
[hash_growth_rehash_drop]        mutant=mutant::HashTable{kSkipRehashOne}            clean=true flagged=true
[aggregation_overflow_i32]       mutant=catalog::mutant::sum_i64_wrapping_i32        clean=true flagged=true  (i32 SUM=-1894967296, true=2400000000)
[selection_vector_aliasing]      mutant=catalog::mutant::gather_in_place_aliased     clean=true flagged=true
[float_to_int_round_plus_half]   mutant=catalog::mutant::cast_f64_to_i64_plus_half   clean=true flagged=true  (engine=-4503599627370498 vs oracle=-4503599627370497)
[join_drop_probe_match]          mutant=mutant::HashJoin{kDropProbeMatch}            clean=true flagged=true  (row count 2 vs 3)
[sort_desc_as_asc]               mutant=mutant::Sort{kDescSortsAsc}                  clean=true flagged=true  (engine=-9 vs oracle=100 at row0)
[agg_fold_null_in_sum]           mutant=mutant::Aggregate{kFoldNullInSum}            clean=true flagged=true  (engine=0 vs oracle=NULL)
[plan_drop_sort]                 mutant=plan::lower_mutant{kDropSort}                clean=true flagged=true  (engine=3 vs oracle=1 at row0)
```

### DuckDB independently validates the carry-forward (reference bypassed)
Per the final review's standing audit lesson ("a planted bug caught *via DuckDB*,
not the circular reference"), I ran the float→int CAST query through **DuckDB alone**
and compared to the `+0.5` mutant on integral doubles ≥ 2^52:
```
duckdb_available=1
row 0: duckdb=4503599627370497   +0.5-mutant=4503599627370498   <-- DuckDB CATCHES the mutant
row 1: duckdb=4503599627370499   +0.5-mutant=4503599627370500   <-- DuckDB CATCHES the mutant
row 2: duckdb=-4503599627370497  +0.5-mutant=-4503599627370498   <-- DuckDB CATCHES the mutant
```
DuckDB returns the correct integer on every row; the mutant diverges on every row.
(228k DuckDB symbols are linked into the meta-test; the differential suites run
DuckDB for real — the 0-row diff takes ~6s and `plan_differential` ~500s under ASan,
not the sub-second reference-only path.)

### Design note — why several check TUs
The per-WP mutant headers `ops/{hashtable,aggregate,join}_mutants.h` each define a
**distinct** `qe::mutant::Mutation` enum, so at most one of that trio may appear in
any single TU. The catalog therefore groups its checks into conflict-free TUs
(`catalog_checks_kernel/hash/join_sort/agg_plan.cpp`); the registry
(`mutation_catalog.cpp`) includes **no** mutant header and only wires the check
functions. At the final link the three enums coexist but never interact (each
mutant's call site and its implementation were compiled from the same header, so
their enumerator values always agree). *Possible future tidy-up (not taken, ICR
candidate): give each family a distinct enum name.*

---

## 2. `compact_column(n==0)` fix — the authorized frozen-core change

**File:** `core/owned_batch.cpp` (body only). **`core/owned_batch.h` is byte-for-byte
unchanged** — the signature `OwnedColumn compact_column(const Column&, const
SelectionVector*, std::size_t)` is preserved; nothing else in `core/` was touched.

**Exact change** (added immediately after `OwnedColumn out = OwnedColumn::make(in.type, n);`):
```cpp
    // n==0 ... return a valid, dense, all-valid EMPTY OwnedColumn of in.type
    // BEFORE the out-of-place assert and the gather (both are no-ops/abort at n==0).
    if (n == 0) return out;
```

**Why it aborted:** at `n==0`, `OwnedColumn::make` allocates a 0-byte `Buffer` whose
`data()` is `nullptr`; an *empty input* column's `data` is likewise `nullptr`, so the
guard `assert(out.data() != in.data && "compaction must be out-of-place")` evaluated
`nullptr != nullptr == false` and fired `SIGABRT`. The gather/validity loops were
already no-ops at `n==0`; only the assert stood in the way.

**Before/after proof (verified):** with the guard removed, `compact_empty_test`
**SIGABRTs** at the empty-input case (`compact_empty_test.cpp:60`,
`owned_batch.cpp:111`); with the guard, it passes **108 assertions**. (Under
`NDEBUG`/release the assert is compiled out, so the abort is a debug/sanitizer-build
crash — the ASan gate is exactly where it bites.)

This is a **WP-15 pinned-bug candidate** (carry-forward (1)). The 0-row corner is
also covered at the *differential* level by `tests/oracle_zero_row_test.cpp` (filter-
selects-nothing, global aggregate over empty input, keyed aggregate over empty,
empty ORDER BY, empty INNER join), each diffed vs DuckDB.

---

## 3. Seeded generators + replay

The seeded schema/data/query/plan generators (`oracle/generators.*`) and their
plan-surface coverage (filter / project / group-by / join / sort / composite
pipelines) feed the differential vs DuckDB. Every randomized test prints its seed
and replays from a single command (`tests/wp1_test_main.cpp`):

```
./build-asan/plan_differential_test --seed 20260614      # any fuzz failure replays from its seed
```

The catalog meta-test itself is **deterministic** (fixed scenarios / a fixed catalog
seed) so it can never flake; the seed it prints is for harness uniformity.

---

## 4. Cross-ISA

No hardcoded vector width / ISA / cache / core-count introduced (the new code is
oracle/test glue + pure scalar mutants). The framework runs unchanged on ARM (here)
and x86 (M5); nothing x86-specific creeps in. Forbidden-include gate clean.

---

## 5. Rigor gates — exact commands

```bash
# Build (DuckDB amalgamation staged at third_party/duckdb/ => QE_WITH_DUCKDB ON)
cmake --preset asan
cmake --build build-asan -j8

# ASan + UBSan, all mutation + WP-9 tests  (asan preset = -fsanitize=address,undefined)
ctest --test-dir build-asan -R \
  'compact_empty_test|oracle_zero_row_test|mutation_catalog_meta_test|.*_mutation_test' \
  --output-on-failure
#  -> 100% (11/11) passed

# The catalog meta-test alone (every mutant flagged; un-mutated passes; vs DuckDB)
./build-asan/mutation_catalog_meta_test --seed 20260614

# Differential vs authoritative DuckDB (QE_WITH_DUCKDB ON, 228k duckdb symbols)
nm build-asan/mutation_catalog_meta_test | grep -c -i duckdb     # -> 228207
./build-asan/oracle_zero_row_test --seed 20260614

# Forbidden-include gate
bash scripts/check_forbidden_includes.sh                          # -> clean

# compact_column before/after regression (proof it bites)
./build-asan/compact_empty_test                                   # passes; SIGABRTs if the guard is reverted
```

Determinism: `QE_CI_SEED=20260614` drives the randomized WP-9 tests in CI (wired in
CMake, same as every other randomized suite).

**Sanitizer status:** WP-9 + all mutation suites **green under ASan+UBSan** (11/11
re-run, above). **Full-tree ASan+UBSan re-run: `100% tests passed, 0 failed out of
37`** (`ctest --test-dir build-asan`, total 502s; includes every differential,
mutation, and bench self-test). Release build of the WP-9 targets also passes
(`build/` preset).

---

## 6. Files

New: `tests/mutation_catalog.{h,cpp}`, `tests/catalog_checks.h`,
`tests/catalog_check_util.h`, `tests/catalog_checks_kernel.cpp`,
`tests/catalog_checks_hash.cpp`, `tests/catalog_checks_join_sort.cpp`,
`tests/catalog_checks_agg_plan.cpp`, `tests/catalog_mutants.{h,cpp}`,
`tests/mutation_catalog_meta_test.cpp`, `tests/compact_empty_test.cpp`,
`tests/oracle_zero_row_test.cpp`, `MUTATION_CATALOG.md`, `WP-9-REPORT.md`.
Changed: `core/owned_batch.cpp` (authorized n==0 fix, body only),
`CMakeLists.txt` (appended WP-9 section only — WP-10's section untouched).

---

## 7. ICRs / message for the final review

- **No ICR.** The only frozen-file edit is the authorized `compact_column` body fix;
  `core/owned_batch.h` is unchanged (verify: `git diff --stat core/owned_batch.h`
  empty).
- **Change-log entry to record:** `core/owned_batch.cpp::compact_column` now returns
  a valid empty `OwnedColumn` on `n==0` (was a `SIGABRT` via the out-of-place
  assert). Signature-preserving; carry-forward (1) closed; WP-15 pinned-bug
  candidate; regression `tests/compact_empty_test.cpp` (+ differential
  `tests/oracle_zero_row_test.cpp`).
- **Carry-forward (3) closed:** "float→int round via +0.5" is now catalog entry
  `float_to_int_round_plus_half`, caught by the project-CAST differential vs DuckDB.
- **Benign multi-`Mutation`-enum coexistence** in `mutation_catalog_meta_test`
  (see §1 design note) — flagged for awareness; an optional future tidy-up is to
  rename the per-family enums (ICR candidate, not needed for correctness).
- **CMake:** I appended a clearly-marked WP-9 section at the end of `CMakeLists.txt`
  and did **not** touch WP-10's section. Both workers append; no overlap.
- The WP-3 `-Wno-error` ratification (carry-forward (2)) and the unused-include nits
  (carry-forward (5)/(7)) are unrelated to WP-9 and left as-is.
