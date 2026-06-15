# Mutation Catalog (WP-9)

> *"A checker that cannot fail proves nothing."* — RIGOR.md rule 4

The differential oracle is only credible because we can enumerate deliberately
broken engine variants and **show the suite flags every one** while the
un-mutated engine passes the same diffs. This file is the documented inventory;
the executable registry is `tests/mutation_catalog.{h,cpp}`, and the gate that
runs every entry is `tests/mutation_catalog_meta_test.cpp`
(`ctest -R mutation_catalog_meta_test`).

The meta-test asserts: (1) **≥ 6** entries (`kMinCatalogEntries`); (2) every named
§12 hazard class is **spanned**; (3) for **every** entry the un-mutated path
passes *and* the mutant is flagged. When `QE_WITH_DUCKDB` is set, every
differential-backed entry escalates to the **authoritative DuckDB** golden model,
so "flagged"/"passes" is decided against DuckDB, not just the reference oracle.

## Executable registry (each is asserted by the meta-test)

| # | id | §12 hazard | Planted mutant | Checker / catching test |
|---|----|-----------|----------------|-------------------------|
| 1 | `simd_tail_remainder` | SIMD tail/remainder past the last full vector | `simd::mutant::all_valid_vec_tailbug` | `all_valid` kernel vs ground truth (`validity_mutation_test`) |
| 2 | `null_propagation_kleene` | three-valued (Kleene) null propagation | `expr::mutant::logic_and_twovalued` | AND truth table vs scalar twin (`expr_mutation_test`) |
| 3 | `all_valid_fastpath_skips_null` | all-valid fast path skipping a genuine null | `expr::mutant::propagate_nulls_and_allvalidbug` | null-propagation combine (`expr_mutation_test`) |
| 4 | `hashjoin_probe_overflow` | hash probe walk overflow / collision storm | `mutant::HashTable{kFindStopEarly}` | unordered cross-check (`hashtable_mutation_test`) |
| 5 | `hash_growth_rehash_drop` | hash capacity-growth rehash | `mutant::HashTable{kSkipRehashOne}` | unordered cross-check (`hashtable_mutation_test`) |
| 6 | `aggregation_overflow_i32` | aggregation integer (accumulator) overflow | `catalog::mutant::sum_i64_wrapping_i32` | global-SUM differential vs DuckDB (meta-test) |
| 7 | `selection_vector_aliasing` | selection-vector index aliasing after compaction | `catalog::mutant::gather_in_place_aliased` | vs out-of-place `compact_column` (meta-test) |
| 8 | `float_to_int_round_plus_half` | float→int cast round via +0.5 — **WP-2 carry-forward** | `catalog::mutant::cast_f64_to_i64_plus_half` | project-CAST differential vs DuckDB (meta-test) |
| 9 | `join_drop_probe_match` | join orchestration: dropped probe match | `mutant::HashJoin{kDropProbeMatch}` | join differential vs DuckDB (`join_mutation_test`) |
| 10 | `sort_desc_as_asc` | sort orchestration: DESC sorted ASC | `mutant::Sort{kDescSortsAsc}` | **positional** differential vs DuckDB (`sort_mutation_test`) |
| 11 | `agg_fold_null_in_sum` | aggregation orchestration: NULL folded as 0 in SUM | `mutant::Aggregate{kFoldNullInSum}` | group-by differential vs DuckDB (`agg_mutation_test`) |
| 12 | `plan_drop_sort` | plan-lowering orchestration: dropped Sort | `plan::lower_mutant{kDropSort}` | **positional** plan differential vs DuckDB (`plan_mutation_test`) |

Entries 6–8 are the **new WP-9 mutants** (`tests/catalog_mutants.*`); entry 8 is
the WP-2 carry-forward ("float→int round via +0.5"). Entries 1–5 and 9–12
consolidate the per-WP mutants the earlier work packages shipped, each still also
exercised by its own catching test (the right-hand column).

## Carry-forward fix (regression, not a survivable mutant)

| Hazard | Fix | Regression |
|--------|-----|-----------|
| `compact_column` **aborts on n==0** (empty input / 0-row selection) — carry-forward (1), WP-15 pinned-bug candidate | `core/owned_batch.cpp::compact_column` returns a valid empty `OwnedColumn` when `n==0`, **before** the out-of-place assert (signature unchanged) | `tests/compact_empty_test.cpp` (aborts before the fix, passes after) + the 0-row differential cases `tests/oracle_zero_row_test.cpp` |

This is recorded as a fix rather than a registry mutant because it requires two
*versions* of `compact_column` linked at once, which is impossible; the
before/after proof is the regression test (verified to SIGABRT before the
one-line guard, pass after).

## Full per-WP mutant inventory (each driven by its own catching test)

The registry above picks a representative mutant per family for the meta-test; the
complete planted-mutant set each per-WP test drives:

- **simd** (`simd/validity_kernels_mutants.*`): `count_set_vec_tailbug`,
  `all_valid_vec_tailbug` — `validity_mutation_test`.
- **expr** (`expr/expr_kernels_mutants.*`): `cmp_lt_i32_tailbug_vec`,
  `logic_and_twovalued`, `propagate_nulls_and_allvalidbug` — `expr_mutation_test`.
- **hashtable** (`ops/hashtable_mutants.*`): `kSkipRehashOne`, `kFindStopEarly`
  — `hashtable_mutation_test`.
- **aggregate** (`ops/aggregate_mutants.*`): `kFoldNullInSum`, `kEmptyGroupZero`,
  `kGroupTailOffByOne` — `agg_mutation_test`.
- **join** (`ops/join_mutants.*`): `kDropProbeMatch`, `kLeftWrongNull`,
  `kCompositeFirstKeyOnly`, `kFanoutTailOffByOne` — `join_mutation_test`.
- **sort** (`ops/sort_mutants.*`): `kDescSortsAsc`, `kNullsFlipped`,
  `kRadixSignBug`, `kUnstableTiebreak`, `kEmitTailOffByOne` — `sort_mutation_test`.
- **plan** (`plan/plan_mutants.*`): `kDropFilter`, `kSwapJoinSides`,
  `kReverseAggKeys`, `kDropSort` — `plan_mutation_test`.
- **operator orchestration** (`tests/oracle_mutation_test.cpp`): inverted filter,
  dropped final partial batch.
- **WP-9 new** (`tests/catalog_mutants.*`): `cast_f64_to_i64_plus_half`,
  `sum_i64_wrapping_i32`, `gather_in_place_aliased`.

## Reproduce

```bash
cmake --preset asan && cmake --build build-asan --target mutation_catalog_meta_test -j8
./build-asan/mutation_catalog_meta_test --seed 20260614        # all 12 flagged; engine passes
```
