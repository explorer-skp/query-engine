# WP-8 Report — Plan + dataframe builder API (closes M2)

**Status:** complete. New module `plan/` (engine-side) + oracle extensions wire the
SAME plan into BOTH the engine operator tree and DuckDB. Builder reproduces every
WP-3..WP-7 query shape plus a deep composite pipeline; all diff green vs the
independent reference AND authoritative DuckDB; printable round-trip stable; a
plan-LOWERING mutation self-test (4 mutants) is caught; ASan/UBSan green; full
suite 32/32, no regression. **No frozen interface was edited (no ICR).**

---

## 1. The `plan/` public surface (frozen at acceptance)

`plan/plan.h` — a small typed IR, one node per frozen operator, plus a fluent
builder. Mirrors `expr::Expr`'s value-handle design.

- **`PlanKind`** — `Scan | Filter | Project | Aggregate | Join | Sort` (1:1 with `ops/`).
- **`Plan`** — immutable, shared (`shared_ptr<const PlanNode>`) value handle.
  - `output_schema()` — derived at build time, no execution.
  - `lower(batch_size = Scan::kDefaultBatchSize)` → `unique_ptr<Operator>` (the live tree).
  - `to_string()` — deterministic, structure-revealing multi-line form (printing only; no re-parse, a non-goal).
  - `kind()`, `node()` (the oracle glue reads node fields).
- **`PlanBuilder`** — fluent: `.filter(expr) .project({{name,expr}}) .aggregate(keys, aggs) .join(build, lkeys, rkeys, type) .sort(keys) .build()`. Each method validates against the current output schema and returns a NEW builder atop the new node. `.plan()` and an implicit `operator const Plan&` expose the assembled plan; `.schema()` exposes the current output schema.
- **`plan::scan(const Table&)`** → `PlanBuilder` (leaf; borrows the Table).
- **`ColRef`** — key reference, index OR name (implicit from `int`/`unsigned`/`string`). **`SortBy`** — `{ColRef, SortDir, NullOrder}` for the name-friendly sort path.

### Column references (the documented split)

- **Expressions** (filter predicate, projection exprs) reference columns **by index**
  into the child schema — the frozen `expr::col(Type, index)` path, matching the
  frozen operator ctors. Name-resolution for an expression is available the
  dataframe way via `expr::col(builder.schema(), "name")` (the builder exposes
  `.schema()`); expression refs are validated against the child schema at build
  time (`validate_expr` checks index-in-range AND recorded-type == schema-type).
- **Key lists** (group/join/sort keys) accept a **`ColRef` = index or name**,
  resolved against the child schema at build time. This is the dataframe-friendly
  path: `.aggregate({"region"}, …)`, `.join(scan(b), {0}, {"id"}, …)`,
  `.sort({SortBy{"n", SortDir::Desc}})`. (Index-based `std::vector<SortKey>` is
  also accepted directly for the frozen path.)

Builder validation throws `std::invalid_argument` on a non-BOOL filter, a bad
index, an unknown name, mismatched/empty join key lists, or a join-key type
mismatch — so a malformed query fails as the tree is assembled, not at run time.

### Table lifetime (the use-after-free hazard, handled)

A `Scan` node **borrows** its `Table` by `const Table*` (mirroring `ops/scan.h` and
`build_engine_pipeline`). The borrowed Table(s) must outlive both the Plan and any
lowered tree. Plans are cheap handles over an immutable shared node, so
copying/moving a Plan never moves a Table. The random generator (`gen_plan_case`)
keeps its Tables in `std::vector<std::unique_ptr<Table>>` (STABLE addresses) so
moving the returned `PlanCase` cannot dangle a Scan pointer — this is the
documented pattern for feeding generated Tables to a Plan.

---

## 2. How the plan unifies the engine-tree and SQL paths

One `Plan` is the single source of truth; both backends derive from it:

- **Engine:** `Plan::lower()` is a mechanical, 1:1 transcription to the operator
  tree — **exactly** the tree the per-WP tests build by hand (same children, same
  order). `lq_to_plan` in the differential test mirrors `build_engine_pipeline`;
  Join lowers `children[0]=probe`, `children[1]=build` (same order as
  `build_join_pipeline`).
- **DuckDB:** `oracle/plan_sql.cpp::render_plan_sql` renders the plan to ONE
  composable SQL query. Each node becomes a `SELECT` over a SUBQUERY whose output
  columns are aliased to canonical positional names `o0..oN`. This handles the two
  hazards a flat renderer can't: (1) join column-name **collisions** (both sides
  may have `c0`), and (2) the expr IR referencing columns **by index** — every
  parent resolves index `i` to the unambiguous `oi`. The read-back uses
  `plan.output_schema()` types. `run_plan_duckdb` (in `duckdb_oracle.cpp`, behind
  `QE_WITH_DUCKDB`) loads the plan's base Tables and runs the SQL.

**Reuse, not duplication (SQL rendering):** the renderer reuses the existing
helpers from `oracle/sql_render.h` — `expr_to_sql`, `sql_type`, and (newly
**exposed**, previously file-local) `agg_call_sql` and `order_by_sql`. I refactored
`order_by_clause` to delegate to the shared `order_by_sql`, so the WP-7 ordinal
rendering is now a single implementation used by both `select_sql` and the plan
renderer. The composition logic (subquery tree, `o`-named columns) is new because
the existing flat `select_sql`/`join_sql` assume a single base table.

---

## 3. Oracle query reps: augmented, not refactored (and why)

I **left `LogicalQuery`/`JoinQuery` in place and added a plan-level path** rather
than rebuilding them on top of the plan. Rationale: (a) zero risk to the green
WP-3..WP-7 differentials — their reps and call sites are byte-for-byte unchanged;
(b) the plan reference *reuses* those reps as its node-level engine, so there is no
duplicated semantics. Specifically:

- **`run_plan_reference`** (independent reference, `reference_oracle.cpp`) walks the
  plan bottom-up, MATERIALIZING each node's output as a `Table` and computing that
  node with the EXISTING references: `run_reference` for
  scan/filter/project/group-by/order-by (expressed as a one-node `LogicalQuery`
  over the materialized child) and `run_join_reference` for join. It shares NO code
  path with `Plan::lower()` (the engine tree under test), so "engine == reference"
  is a meaningful differential of the WHOLE composition. The only new helpers are
  `rs_to_table` (ResultSet→Table, pure cell copy) and `identity_projections`.
- **`run_plan_differential` / `run_plan_vs_reference`** (`differential.cpp`) reuse
  the frozen comparator. A plan whose **root is Sort** diffs POSITIONALLY (ORDERED,
  D12); otherwise canonicalized — picked from the root node kind.

---

## 4. Coverage (the rigor gate)

`tests/plan_differential_test.cpp` — every WP-3..WP-7 shape through the PLAN path
(random, reusing the divergence-safe generators), each diffed vs reference AND
DuckDB across batch sizes {64,256,2048}:

- scan (identity); scan→filter→project; GROUP BY single/composite/global; INNER &
  LEFT join single/composite; ORDER BY multi-col ASC/DESC NULLS FIRST/LAST
  (positional);
- **DEEP composite** `scan→filter→join→aggregate→sort` — `gen_plan_case` (random,
  80 iters) + an explicit composite-key `join→group-by→order-by` case. This is the
  shape no single earlier WP exercised.
- Edge cases: empty table through filter→project; global aggregate; LEFT join with
  unmatched (NULL-filled) probe rows.

`tests/plan_print_test.cpp` — exact canonical rendering, stability (two builds
print identically), structural change visible (dropping Filter / swapping join
sides), name-resolved == index-resolved keys, and validation rejections.

---

## 5. Mutation self-test (oracle is sacred)

`plan/plan_mutants.cpp::lower_mutant` is a deliberately-wrong transcription of
`Plan::lower()` (test-only lib `qe_plan_mutants`, never linked into the engine),
committing ONE plan-LOWERING defect:

| Mutation | Defect | Caught by |
|---|---|---|
| `kDropFilter` | drops a Filter node | row-count mismatch (deep pipeline, LEFT join re-introduces the filtered key) |
| `kSwapJoinSides` | swaps probe/build children + keys | column type/shape mismatch |
| `kReverseAggKeys` | reverses composite GROUP BY keys | key column type mismatch |
| `kDropSort` | drops a Sort node | POSITIONAL value mismatch (root is Sort) |

`tests/plan_mutation_test.cpp` shows each mutant is FLAGGED by the plan differential
(reference + DuckDB) and the correct `Plan::lower()` PASSES the same diff. The
`kDropSort` catch specifically proves the POSITIONAL comparator bites where a
canonicalizing compare would hide it (`value mismatch at row 0 col 0: engine=3
oracle=1`).

---

## 6. Scalar-twin applicability

**Not applicable.** WP-8 is a plan/orchestration layer; it adds NO vectorized
kernel (no Highway, no new SIMD). It composes the frozen operators, which carry
their own scalar twins. The from-scratch grep gate now covers `plan/` and is clean.

---

## 7. Assumptions

- The plan→SQL renderer assumes the divergence-safe generator grammar (the deep
  generator uses only overflow-proof aggregates COUNT/MIN/MAX and an ORDER BY over
  every output column, so the positional diff is total-ordered and DuckDB-exact —
  matching the existing generators' contract). The `DuckDBError` "regenerate, don't
  diff" backstop remains and did not fire.
- 0-row plans do not reach the known `compact_column` n==0 abort: `Scan` returns
  `nullopt` for an empty table so `Filter` never compacts 0 rows, and the reference
  uses `rs_to_table` + the n==0-guarded `run_reference`. Verified under ASan. (This
  matches the carry-forward note for WP-5/6/7.)

## 8. ICRs

**None.** `git diff` shows no change to any frozen header (core/simd/expr/
ops-operator/scan/filter/project/aggregate/join/sort/table). The only oracle-header
edits are additive (new `agg_call_sql`/`order_by_sql` decls in `sql_render.h`; new
`run_plan_*` decls), and `sql_render.cpp`'s `order_by_clause` was refactored to
delegate to the shared renderer with identical output.

---

## 9. Exact commands

All from repo root `/Users/skp/work/query-engine`. DuckDB amalgamation is staged at
`third_party/duckdb/`, so `QE_WITH_DUCKDB` is auto-enabled and DuckDB actually runs.

**Build (release):**
```
cmake --preset release
cmake --build --preset release -j
```
Configure prints: `WP-3 oracle: DuckDB amalgamation FOUND -> DuckDB backend ENABLED`.

**From-scratch gate (now covers `plan/`):**
```
scripts/check_forbidden_includes.sh
# -> clean — no forbidden includes/links in core simd expr ops plan tsx
```

**Printable round-trip:**
```
./build/plan_print_test
# -> 5 cases / 10 assertions, SUCCESS
```

**Plan differential vs BOTH reference and DuckDB (seeded; one-command replay):**
```
./build/plan_differential_test --seed 20260614
# -> [wp1] seed=20260614 ... 10 cases / 765 assertions, SUCCESS
```
Evidence DuckDB genuinely ran (not stale reference-only): in the full release
ctest, `plan_differential_test` took **19.72 s** (reference-only is sub-second);
the binary links the staged amalgamation.

**Mutation self-test (flags the lowering mutant; passes with it reverted):**
```
./build/plan_mutation_test --seed 20260614
# -> 4 cases / 9 assertions, SUCCESS
#    each TEST_CASE: CHECK(correct lower passes) + CHECK_FALSE(mutant) + caught-message
#    e.g. "dropped-sort mutant caught: value mismatch at row 0 col 0: engine=3 oracle=1"
```

**ASan + UBSan (gate; re-run):**
```
cmake --preset asan
cmake --build --preset asan --target plan_print_test plan_differential_test plan_mutation_test -j
./build-asan/plan_print_test --seed 20260614          # SUCCESS, no sanitizer output
./build-asan/plan_mutation_test --seed 20260614        # SUCCESS
./build-asan/plan_differential_test --seed 20260614     # 765 assertions SUCCESS, no sanitizer output
```

**Full regression (no WP-3..WP-7 test regressed):**
```
ctest --preset release -j
# -> 100% tests passed, 32/32 (the 3 new WP-8 tests + all prior)
```

---

## 10. Files

New: `plan/plan.h`, `plan/plan.cpp`, `plan/plan_mutants.h`, `plan/plan_mutants.cpp`,
`oracle/plan_sql.h`, `oracle/plan_sql.cpp`, `tests/plan_print_test.cpp`,
`tests/plan_differential_test.cpp`, `tests/plan_mutation_test.cpp`.

Augmented (additive / non-regressing): `oracle/sql_render.{h,cpp}` (expose
`agg_call_sql`/`order_by_sql`, delegate `order_by_clause`), `oracle/reference_oracle.{h,cpp}`
(`run_plan_reference`), `oracle/differential.{h,cpp}` (`run_plan_*`),
`oracle/duckdb_oracle.{h,cpp}` (`run_plan_duckdb`), `oracle/generators.{h,cpp}`
(`gen_plan_case`/`PlanCase`), `CMakeLists.txt` (`qe_plan`, `qe_plan_mutants`,
`plan_sql.cpp` in `qe_oracle`, 3 tests).
