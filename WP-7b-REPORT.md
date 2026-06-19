# WP-7b — Dictionary-encoded strings (VARCHAR) — Worker Report

## What landed
`Type::STR` = a **dictionary-encoded VARCHAR**: the column's `data` holds **int32
dictionary codes** (`byte_width(STR)==4`); an out-of-band `StringDict` (new helper
header `core/string_dict.h`) resolves code→bytes. **Equality/order is ALWAYS by
string VALUE**, never by raw code — two columns may carry different dictionaries and
the same string can get different codes in each. That asymmetry is the planted
mutant.

## The two pre-authorized ICRs (additive — audit with `git diff core/types.h core/column.h`)
- **ICR-1 `core/types.h`:** appended `STR` to `enum class Type` **after `TS`**
  (`{I32,I64,F64,BOOL,TS,STR}`) and `case Type::STR: return 4;` in `byte_width`.
  Existing enumerators are byte-unchanged (same order/values); only the enum line is
  extended to append STR, exactly as ICR-1 specifies.
- **ICR-2 `core/column.h`:** appended ONE field `const StringDict* dict;` **after
  `all_valid`** (existing fields byte-unchanged) + a forward declaration of
  `StringDict`. `dict == nullptr` for every non-STR column. New sibling helper header
  `core/string_dict.h` (interned, immutable code↔bytes table: contiguous bytes +
  offsets + an `unordered_map` dedup index; `intern`/`at`/`size`; std-only,
  hand-rolled — no string/dataframe library; no ISA/width/cache in its surface).
- `core/buffer.h`, `Batch`, `SelectionVector`, `Schema` UNCHANGED. **No other frozen
  header changed a signature** (simd / `expr.h` / every `ops/*.h` / `plan.h` / `tsx/*`
  / the oracle surfaces) — internals only, extended to handle STR.

## Representation & lifetime (the one real design decision)
`OwnedColumn` gained a `std::shared_ptr<const StringDict>` plus `make_str(len,dict)`,
`set_dict` (owning) / `set_dict_ref` (aliasing, non-owning), and `view()` publishes
`Column::dict`. Pipeline operators **drain their child before emitting**, so a child's
batches (and their dicts) are dead by emit time. Therefore every operator that
outputs STR **owns a canonical `StringDict`** built during the drain (re-interning by
VALUE): Aggregate (group keys + `MIN/MAX(str)`), Sort (materialized columns),
HashJoin build side, and the reference/parallel `rs_to_table`. `scan.cpp` (which
hand-builds the `Column` view) now copies `oc.dict()` through; `compact_column`,
join/sort gathers carry the dict through (codes are width-4, moved by the existing
`gather32` path).

## Equality "by value": canonicalization, and the mutant
The frozen `HashTable` compares `uint64` key words, so STR keys are made
**value-consistent** before they reach it:
- **GROUP BY:** each STR key column is canonicalized into the aggregate's owned dict
  (equal values → equal codes), then fed as STR; read-back code → output dict.
- **JOIN:** build & probe STR keys are interned into ONE **shared value-id space** and
  fed to the table as **I32** ids (the table never sees STR keys). Equal strings on
  the two independently-dictionaried sides get the same id → they match.
- `hash_internal::normalize_value(STR)` reads the int32 code (correct for the
  single-dict group-by path; join canonicalizes upstream so STR never reaches it).
- **Planted mutant (`ops/string_key_mutants.*`, own enum `qe::mutant::StringKeyMutation`):**
  a faithful copy of `HashJoin` reusing the SAME `join_internal.h` plumbing whose only
  defect (`kHashRawCode`) **skips canonicalization and hashes the raw codes** — so
  equal strings with different per-side codes miss every cross-dictionary match. The
  differential catches it; `kNone` (control) and the real operator pass.

## STR-invalid sites — fail cleanly and loudly
- **expr:** `arith(STR,…)` and `cast` to/from STR throw `std::invalid_argument` at
  build time; STR literals throw (the frozen `Scalar` has no string payload, so STR
  comparisons are **column-vs-column**). Comparison `= != < <= > >=` on STR is
  lexicographic byte compare via the dicts.
- **aggregate:** `agg_result_type` throws for `SUM/AVG(STR)`; `MIN/MAX(str)→STR`,
  `COUNT` works.
- **sort:** STR never takes the radix fast-path (`radix_eligible` already excludes it);
  comparison path orders by resolved bytes.
- **tsx (asof/window/compress):** STR is out of their grammar → clean
  unsupported-type asserts (no STR timestamps / Gorilla).
- The `-Wswitch -Werror` sweep across ~30 type-switch sites is what enumerated every
  one of these; STR-invalid cases are loud, STR-valid cases implemented.

## scalar==vector decision
The STR-specific logic (hash/compare/min-max **by value**) is **inherently scalar**
(variable-length byte compare) — there is **no Highway twin**, and the Vector and
Scalar expr backends drive the *same* `cmp_str` routine (the WP-5/WP-6/WP-12
precedent for inherently-sequential steps; documented in `expr/eval.cpp`). The only
vectorized step STR touches is the **fixed-width int32 code GATHER** (compact/join/
sort), which is the EXISTING `gather32` kernel already covered by the WP-1/WP-6/WP-7
`scalar==vector` tests. Hence **no new `scalar==vector` target** — by design, stated
per the brief.

## Oracle wiring (the differential)
- **generators:** `gen_string_column(rng, alphabet, n, null_pct)` (public, additive) —
  short ASCII strings (no NULs), FRESH per-column dict so equal values get different
  codes; `gen_column` is STR-capable. (The existing random suites were left numeric to
  preserve their determinism; a **dedicated** WP-7b differential drives the string
  scenarios deterministically.)
- **sql_render:** `STR → VARCHAR`. **duckdb_oracle:** INSERT decodes code→quoted text
  (doubling quotes); results read VARCHAR into `Cell.s`.
- **reference_oracle:** STR handled by `std::string` value — `RefKey` carries the
  string so group/join keys compare by value; `MIN/MAX(str)` by `std::string`;
  `apply_order_by` lexicographic; `rs_to_table` rebuilds a dict.
- **result_set:** `Cell` gained `std::string s`; STR compared by decoded value
  (EXACT, no float tolerance); canonicalization sorts on the strings.

## Mutation self-test + catalog (rule 4)
- `tests/wp7b_strings_mutation_test.cpp` — real passes; `kHashRawCode` **CAUGHT**
  (`CHECK_FALSE`, non-vacuous) on a cross-dictionary join.
- `tests/catalog_checks_strings.cpp` + ONE append-only `Entry` in
  `tests/mutation_catalog.cpp` (`string_key_by_code`, hazard
  `Hazard::kOperatorOrchestration` — no new `Hazard` enumerator). Wired into
  `mutation_catalog_meta_test` (now 16 entries, all flagged).

## CMake
New: `core/string_dict.h` (header-only, joins `qe_core` by inclusion); new
`qe_strings_mutants` lib (`ops/string_key_mutants.cpp`); new tests
`wp7b_strings_differential_test`, `wp7b_strings_mutation_test`
(each `--seed ${QE_CI_SEED}`); `catalog_checks_strings.cpp` + `qe_strings_mutants`
added to `mutation_catalog_meta_test`. Forbidden-include gate clean over all dirs.
No `scalar==vector` target (STR path is scalar, above).

## Commands (reproduce)
```
# configure + build (ASan+UBSan combined preset) with DuckDB staged at third_party/duckdb
cmake --preset asan && cmake --build build-asan -j8

# the DuckDB string differential (group-by / join / filter / order-by / MIN-MAX)
./build-asan/wp7b_strings_differential_test --seed 20260614
# the mutation self-test (real passes; kHashRawCode caught)
./build-asan/wp7b_strings_mutation_test --seed 20260614
# the catalog meta-test (string_key_by_code: clean_passes && mutant_flagged)
./build-asan/mutation_catalog_meta_test --seed 20260614

# audit: additive-only frozen diff + from-scratch gate
git diff core/types.h core/column.h
bash scripts/check_forbidden_includes.sh
```

## Results (host=mac-m, isa=neon — correctness only; no perf claim)
- `wp7b_strings_differential_test`: **6 cases / 631 assertions pass** vs the independent
  reference AND **vs DuckDB VARCHAR** (`QE_WITH_DUCKDB` on; binary carries 227,745
  DuckDB symbols, so DuckDB genuinely ran).
- `wp7b_strings_mutation_test`: real passes; `kHashRawCode` caught
  (`value mismatch … engine='GOOG' oracle='AAPL'`).
- `mutation_catalog_meta_test`: 16/16 entries, all `clean_passes && mutant_flagged`.
- ASan/UBSan: green (build-asan preset is ASan+UBSan combined).
- `git diff core/types.h core/column.h`: additions-only (enumerators/fields
  byte-unchanged). Forbidden-include gate: clean.

## Assumptions / notes for the final review
1. **Shared-tree contention:** WP-10b's work was already in the tree. I made only
   additive edits to the three shared files (`CMakeLists.txt`, `tests/catalog_checks.h`,
   `tests/mutation_catalog.cpp`) — appended after WP-10b's entries, none of theirs
   touched — and added the necessary `case Type::STR` to WP-10b's
   `exec/parallel.cpp::result_set_to_table` (mirrors `rs_to_table`; required for the
   shared tree to build with `Type::STR`). The `scripts/check_forbidden_includes.sh`
   change in `git status` is WP-10b's, not mine.
2. **No new ICR.** Only ICR-1/ICR-2 were used.
3. The existing random generators were deliberately left numeric (so the existing
   differential suites' determinism — esp. sort tie-ordering — is unperturbed); STR is
   exercised by the dedicated WP-7b differential. If you want STR folded into the
   broad random suites too, adding `STR` to `rand_type` is safe per analysis (projections
   pass STR through; group-by MIN/MAX/COUNT/keys handle STR; the total-order ORDER-BY
   tiebreaker design covers STR ties) but I left that call to you.
```
