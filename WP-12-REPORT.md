# WP-12 — Backward As-Of Join (Phase-2 headline)

**Module:** `tsx/asof.{h,cpp}` (new). Conforms to the frozen `Operator` contract
(`ops/operator.h`); `tsx/asof.h` is WP-12's public surface (to be frozen at
acceptance — WP-13/WP-15 may reference it).

**Status:** differential GREEN vs the independent reference **and** authoritative
DuckDB v1.1.3 `ASOF JOIN` (`QE_WITH_DUCKDB` staged, DuckDB actually runs);
scalar==vector GREEN; 4 planted mutants caught + 1 consolidated-catalog entry;
ASan+UBSan green (re-run); forbidden-include gate clean (now covers `tsx/`).

---

## 1. Public surface (`tsx/asof.h`) + output column order

```cpp
namespace qe::tsx {
enum class AsofType { Inner, Left };
class AsofJoin : public Operator {
  static constexpr std::size_t kOutBatch = 2048;
  AsofJoin(unique_ptr<Operator> probe, unique_ptr<Operator> build,
           vector<uint32_t> probe_keys, vector<uint32_t> build_keys,
           uint32_t probe_time, uint32_t build_time, AsofType type,
           optional<int64_t> tolerance = nullopt);
  void set_paths(HashPath, GatherPath);   // test seam (defaults kVector/kVector)
  // open()/next()/close()/output_schema() — the frozen contract.
};
}
```

* **Keys:** 0+ equality partition-key columns (equal length, positionally-matching
  types). **0 keys = a single global partition** (verified vs DuckDB; the frozen
  `HashTable` requires ≥1 key, so the zero-key case skips the table — one global
  group id 0).
* **Ordering column:** exactly one timestamp column per side, `Type` `TS` / `I32` /
  `I64` (read as `int64`).
* **Output schema / column order (documented contract, same as `HashJoin`):** ALL
  probe columns in child order, then ALL build columns in child order. Names are
  the children's verbatim and may collide across sides — consumers use POSITION.

`GatherPath` and `HashPath` are reused from `ops/join.h` / `ops/hashtable.h` (no new
enums); `set_paths` is the scalar==vector test seam (downstream WPs use defaults).

## 2. Sorted-merge-per-key-group design (D10 / §5)

Reuses the **frozen pieces** rather than reimplementing them:

1. **Sort** both inputs by `(key cols…, timestamp)` ascending via the frozen
   `ops/sort.h` `Sort` operator. Build groups then arrive timestamp-ascending; a
   probe key's rows form one contiguous, timestamp-ascending run.
2. **Partition by key** via the frozen `ops/hashtable.h` `HashTable`
   (`NullPolicy::kNeverMatch`): each build row gets a stable group id and each
   group keeps its build rows in timestamp order; a probe row's group id is found
   *without* replicating the sort's key comparator (the clean way to align groups
   across the two sorted sides). Zero-key case bypasses the table.
3. **Merge** per key group with a **per-key advancing cursor**: walking a group's
   probe rows in ascending timestamp, a build cursor only moves forward to the last
   `tb <= t` (the nearest preceding) — `O(n+m)` per group. The cursor + current
   group id persist across probe batches (a group's run can straddle a batch
   boundary). This merge is sequential scalar control flow (the WP-6 probe-walk /
   WP-5 grouped-scatter precedent — fine to be scalar; see §6).
4. **Materialize** the matched build row + carried probe row through the WP-1 gather
   kernels (`simd/gather_kernels.h`) via the shared `ops/join_internal.h` emitters
   (`BuildStore` + `emit_probe_column` / `emit_build_column`) — reused **unchanged**
   from WP-6. Output is emitted in dense ≤2048-row batches; a LEFT unmatched probe
   row carries the `kNullBuildRow` sentinel.

## 3. DuckDB `ASOF JOIN` semantics matched (verified, not assumed)

Verified against the staged DuckDB v1.1.3 amalgamation (probe programs +
the live differential):

* **Boundary is `>=` (inclusive):** the match is the GREATEST `tb <= t`, so a build
  timestamp EQUAL to the probe timestamp DOES match. (Using `<` / `>` drops
  equal-timestamp matches — the classic as-of bug; it is mutation
  `kBoundaryStrict`, caught by the suite.)
* **INNER** drops unmatched probe rows; **LEFT** emits them once with all build
  columns NULL.
* **Ties / determinism:** the generators make build-side `(key, timestamp)` UNIQUE
  by emitting **globally-distinct build timestamps**, so the nearest-preceding pick
  is unambiguous across engine / reference / DuckDB (no flaky diff). The operator
  itself does not promise a tie-break (DuckDB does not define one either).

### NULL handling — two **documented divergences** (constrained out of the grammar)

The brief asked to "confirm DuckDB's behavior and mirror it." Confirming it
empirically showed DuckDB v1.1.3's ASOF NULL behavior is **not a clean
never-match**, so — exactly like the div-by-zero / overflow divergences already in
`oracle/generators.h` — we keep the engine on the **principled "NULL never matches"**
path and **constrain generation** to keep the differential honest, validating the
engine's NULL behavior against the independent reference directly (see the
`asof_differential_test` "NULL keys & timestamps" edge, which is reference-only):

1. **NULL timestamp:** DuckDB matches a NULL *probe* timestamp to a NULL *build*
   timestamp within the same partition (`NULL >= NULL` treated as a self-equal
   group). The engine treats any NULL timestamp as no-match.
2. **NULL key:** DuckDB's ASOF NULL-key matching is **data-dependent** — a probe
   with a NULL key component matches a NULL-key build row when the full table is
   present, but NOT when that probe row is queried in isolation (a hash-partition
   artifact; reproduced both ways). There is no well-defined semantics to mirror.
   (A NULL key correctly never matches in a regular hash join — WP-6 — which is
   *not* data-dependent.)

NULL keys/timestamps are therefore excluded from the DuckDB-backed random grammar;
the engine's principled never-match is checked engine-vs-reference. This is the
established `generators.h` divergence-handling pattern, not a gap.

## 4. Within-tolerance variant

Semantics = pandas `merge_asof(tolerance=)` / kdb `aj` window: find the nearest
PRECEDING build row, then DROP it if `t - tb > tolerance` (treated as no-match:
INNER drops, LEFT NULL-fills). Because the nearest preceding `tb` is the closest
from below, "nearest then check" and "nearest within the window" coincide for a
backward as-of. `tolerance` is in the timestamp column's integer units.

**DuckDB rendering (verified equivalent):** INNER adds the window as an extra ON
conjunct `(p.t - b.t) <= tol` (DuckDB correctly drops out-of-window rows). LEFT
keeps the plain `ASOF LEFT JOIN` and NULL-MASKS each build column with
`CASE WHEN (p.t - b.t) <= tol THEN b.col END` — because the extra-ON-conjunct form
would wrongly DROP the probe row in a LEFT join (confirmed). Both INNER+tol and
LEFT+tol diff GREEN vs DuckDB in the random grammar.

## 5. Plan extension is additive-only

`PlanKind::AsofJoin` + dedicated `PlanNode` fields (`asof_left_keys`,
`asof_right_keys`, `asof_left_time`, `asof_right_time`, `asof_type`,
`asof_tolerance`) + `PlanBuilder::asof_join(...)` + lowering + printable form +
`oracle/plan_sql.cpp` `render_asof`. **`git diff` confirms additions only:**
`plan.cpp` and `plan_sql.cpp` have **zero** deletions; `plan.h`'s only modified
existing line is the single-line `PlanKind` enum, to which `, AsofJoin` is appended
— the six existing enumerators keep their spelling, order, and therefore values.
No existing `PlanNode` field or builder signature changed. **No ICR required.**

The new enumerator mechanically forces a new arm in two exhaustive non-authorized
switches (`reference_oracle.cpp::eval_rs`, `plan_mutants.cpp::lower_mutant`); both
arms are appended faithfully (existing arms byte-unchanged) — the expected
consequence of an additive enum, not a frozen-interface edit. `eval_rs` lowering of
`AsofJoin` calls the new `run_asof_reference`, so the plan reference path works
end-to-end for as-of too.

## 6. Scalar-twin applicability

The per-key MERGE is sequential scalar control flow (no vector twin — the same
precedent as the WP-6 probe walk). The vectorized step under test is the column
**GATHER** (matched build row + carried probe row), which reuses the WP-1
`simd/gather_kernels.h` vector kernel + its independently-written scalar twin via
the `join_internal.h` emitters. `tests/asof_scalar_vector_test.cpp` forces
`GatherPath` kVector vs kScalar (and `HashPath` kVector vs kScalar for the
partition hash) on identical input and asserts **byte-identical** output
(300 assertions GREEN). On Apple M5/NEON the gather is not the as-of hot path
(see §8), but the scalar==vector equivalence is what the rule-3/D17 check proves.

## 7. ICRs

None. The plan extension used only the final review-authorized additive growth.

---

## 8. Relative perf (preliminary / relative-only — RIGOR.md §2)

`host=mac-m5 isa=neon` — **Mac numbers are preliminary / relative-only**; no
headline / absolute / AVX-512 claim from macOS.

```
as-of: N=400000 build & probe, key card=2000, (4 build + 3 probe) I64/TS cols, INNER
  vec-gather   = 75.9 ms   (~5.3M probe-rows/s end-to-end incl. sort + merge)
  scalar-gather= 75.2 ms
  vec/scalar gather ratio ≈ 0.99x
```

Honest finding: the as-of operator is dominated by the two sorts + the per-key
merge, **not** the gather (a 64-bit gather of a few columns is memory-bound), so the
vec-vs-scalar gather ratio is ≈1.0x here — the gather is correct-and-equal across
paths but not the hot path for as-of. Reproduce: see "commands" below.

---

## 9. Files

* **New:** `tsx/asof.{h,cpp}`, `tsx/asof_mutants.{h,cpp}`,
  `tests/asof_differential_test.cpp`, `tests/asof_scalar_vector_test.cpp`,
  `tests/asof_mutation_test.cpp`, `tests/catalog_checks_asof.cpp`.
* **Additive edits:** `plan/plan.{h,cpp}`, `oracle/plan_sql.cpp` (authorized);
  `oracle/reference_oracle.{h,cpp}` (`run_asof_reference` + `eval_rs` arm),
  `oracle/generators.{h,cpp}` (`AsofCase` + `gen_asof_case`),
  `plan/plan_mutants.cpp` (`lower_mutant` arm), `tests/catalog_checks.h` +
  `tests/mutation_catalog.cpp` (appended entry #13), `MUTATION_CATALOG.md`,
  `CMakeLists.txt` (`qe_tsx`, `qe_asof_mutants`, three test targets, meta-test wiring).

## 10. Commands (every number/claim reproduces)

```bash
# Build (Release) + sanitizers
cmake --preset release && cmake --build --preset release -j
cmake --preset asan    && cmake --build --preset asan -j        # ASan + UBSan

# Forbidden-include gate (tsx/ is covered)
scripts/check_forbidden_includes.sh

# As-of differential vs reference AND DuckDB (QE_WITH_DUCKDB auto-on; DuckDB runs)
./build/asof_differential_test --seed 12345        # 9 cases / 216 assertions GREEN
ctest --preset release -R asof                     # all three asof tests

# scalar == vector (gather + partition hash)
./build/asof_scalar_vector_test --seed 12345       # 300 assertions GREEN

# Mutation self-test (real passes; every mutant flagged) + consolidated catalog
./build/asof_mutation_test --seed 12345
./build/mutation_catalog_meta_test --seed 12345    # incl. entry 13 asof_boundary_strict

# Sanitizers (re-run; macOS has no LeakSan, ASan+UBSan active)
ctest --preset asan                                # 40/40 GREEN

# Replay any fuzz failure from its printed seed:
./build/asof_differential_test --seed N
```

### Mutation self-test output (real passes, every mutant flagged)

```
./build/asof_mutation_test --seed 12345
MUTATION: the REAL as-of join passes the differential .......... OK
MUTATION: kBoundaryStrict (drops equal-timestamp match) is CAUGHT
MUTATION: kNearestFollowing (forward instead of backward) is CAUGHT
   kNearestFollowing caught: value mismatch at canonical row 0 col 3: engine=30 oracle=20
MUTATION: kIgnoreLastKey (composite over-match) is CAUGHT
   kIgnoreLastKey caught: row count mismatch: engine=1 oracle=0
MUTATION: kLeftWrongNull (unmatched emits real build row) is CAUGHT
   kLeftWrongNull caught: null mismatch at canonical row 0 col 2: engine=1 oracle=NULL
[doctest] test cases: 5 | 5 passed | 0 failed
```
