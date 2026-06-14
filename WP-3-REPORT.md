# WP-3 Report — Scan + Filter + Project, and the first DuckDB oracle wiring

Milestone **M1**: the first end-to-end `scan → filter → project` pipeline,
validated by a differential oracle on seeded random data, with a mutation
self-test proving the oracle bites. On acceptance this WP **freezes
`ops/operator.h`**.

---

## 1. The Operator contract I froze (`ops/operator.h`)

Kept exactly the DRAFT shape — minimal, four pure-virtuals — and froze it:

```cpp
class Operator {
  virtual void open() = 0;
  virtual std::optional<Batch> next() = 0;   // nullopt => exhausted
  virtual void close() = 0;
  virtual Schema output_schema() const = 0;
  virtual ~Operator() = default;
};
```

Frozen semantics now documented in the header (the contract every later operator
obeys):
- **`next()` returns a VIEW** whose bytes are owned by the operator (typically an
  `OwnedBatch` member) and are valid **only until the next `next()`/`close()`**.
  The caller consumes/copies before pulling again (standard pull model).
- **No empty batches:** a returned batch has `row_count >= 1`; an operator that
  would yield zero rows pulls again or returns `nullopt`. After the first
  `nullopt`, `next()` keeps returning `nullopt`.
- `open()` once before the first `next()`; an operator opens its own children.
- `output_schema()` is valid after construction (no `open()` required).
- Width/ISA/cache/core never appear here (Mac→x86 rule). No additions without an
  ICR; per-operator constructors/state are the operator's own.

## 2. The table / source shape (`ops/table.h`)

A `Table` is deliberately minimal and built entirely on the WP-1 owning layer: a
`Schema` (named, typed fields) + one `OwnedColumn` per field, all equal length. It
**owns** its column storage; Scan hands out Column views into it (zero-copy), so a
Table must outlive any operator tree scanning it. `full_batch_view()` (a dense
whole-table Batch) exists for the reference oracle. WP-8's plan layer will drive
Table construction later; the shape is intentionally unopinionated.

**Scan** yields dense ≤`batch_size`-row batches as **zero-copy** views.
`batch_size` defaults to 2048 (D4) and is required to be a positive multiple of
**64** (enforced by a thrown `std::invalid_argument`, so it holds in release too).
Reason: the validity bitmap packs 64 values per word, so a batch starting at row
`start` can expose its validity as a whole-word pointer offset (`start/64`) only
when `start` is a multiple of 64 — which every batch start is when `batch_size %
64 == 0`. The 64 is the format's validity **word** width (from `core/validity.h`),
**not** an ISA/vector width, so there is no Mac→x86 hazard. The **tail** batch may
have a non-multiple-of-64 row count; that only shrinks its own `len`, and bits
past `len` are padding the contract already forbids reading. Tested at 0, 1, 63,
64, 65, 200, 5000 rows and batch sizes {64, 256, 2048}.

**Filter** evaluates the BOOL predicate via `expr::evaluate` (which applies the
input batch's selection vector and returns a dense column), then **compacts** the
passing rows into a fresh dense `OwnedBatch` (D6 first cut). Key correctness
points, all tested:
- **NULL predicate ⇒ row does not pass** (SQL WHERE semantics).
- **Input selection vector handled**: passing *physical* indices are
  `sel_at(b.sel, k)`; compaction reads through those. `compact_column` is
  out-of-place, so the §12 selection-vector aliasing hazard cannot arise.
- Never emits an empty batch (skips all-fail batches; `nullopt` at end).

**Project** evaluates a list of named expr trees per batch into a fresh dense
`OwnedBatch` (the SELECT list); `output_schema()` is `(name, expr.type())`.

## 3. Float epsilons (D11) and canonicalization (D12)

- **Integer-family columns (I32/I64/BOOL/TS): exact equality.** TS is compared as
  its underlying int64 ns (mapped to DuckDB `BIGINT`), so it is exact and we avoid
  DuckDB's microsecond temporal truncation; no temporal functions are exercised
  in WP-3.
- **F64 columns: relative + absolute epsilon** —
  `|a-b| <= kAbsEps + kRelEps*max(|a|,|b|)` with **`kAbsEps = kRelEps = 1e-9`**
  (`oracle/result_set.h`). WP-3 projections are element-wise (no summation
  reorder), so engine-vs-reference floats are bit-identical and engine-vs-DuckDB
  are at worst a rounding step apart; 1e-9 is comfortably safe yet tight. (WP-5's
  reordered float *aggregates* will need a looser, documented epsilon — out of
  scope here.)
- **D12:** WP-3 queries have no ORDER BY, so the comparator **sorts both result
  sets on all output columns** before diffing (NULLs sort first, then by value;
  total order). Nullness is compared exactly on every cell.

## 4. How I handled the ORACLE DIVERGENCE

`expr/expr.h` documents cases where the engine yields **NULL** but DuckDB
**raises**: div/mod-by-zero, out-of-range/non-finite float→int casts, and signed
integer overflow on `+,-,*`. Since DuckDB raises at the **statement** level (the
whole query fails, not per-row), there is no clean per-row equivalence to define.
I therefore chose **option (a): constrain generation so these cases never arise**,
so DuckDB never raises and there is nothing to reconcile (`oracle/generators.h`
documents this in full):
- Integer data and arithmetic are **magnitude-bounded** (`kI32Abs=30000`,
  `kI64Abs=1e9`, multipliers `≤100`, depth-limited) so `+,-,*` cannot overflow.
- **No division/modulo** is generated (also sidesteps DuckDB's `/` float-division
  vs `//` integer-division mismatch). The engine and expr layer fully support
  them and `sql_render` emits the correct `//`/`/`/`%`; they are simply outside
  the first oracle's grammar.
- **Casts are widening only** (I32→I64, I32→F64, I64→F64) — never lossy/OOR.
- **F64 data is finite and moderate** (no inf/NaN) — no float→int OOR, no NaN
  canonicalization ambiguity.

Defense in depth: the DuckDB backend treats any statement error as
`DuckDBError` → the harness **skips/regenerates** that case rather than reporting
a diff. Given the constraints it should never fire.

## 5. DuckDB version pinned + staging requirement (NETWORK-BLOCKED)

The sandbox blocks network (`curl`/`wget` denied) and no DuckDB is present, so I
**could not fetch or run DuckDB myself**. The DuckDB backend is fully written and
wired; CMake auto-enables it when the amalgamation is staged. **Exact artifact the
human must pre-stage** (also in `third_party/duckdb/VENDORING.md`):

- **DuckDB v1.1.3**, asset **`libduckdb-src.zip`**
- **`https://github.com/duckdb/duckdb/releases/download/v1.1.3/libduckdb-src.zip`**
- unzip `duckdb.hpp`, `duckdb.h`, `duckdb.cpp` into `third_party/duckdb/`

CMake then defines `QE_WITH_DUCKDB`, compiles the amalgamation into a test-only
`duckdb_amalg` lib, and `oracle_differential_test` / `oracle_mutation_test`
additionally diff against DuckDB through the **identical** comparator. The backend
uses only the stable `Connection::Query()` + `MaterializedQueryResult` + `Value`
API, so it should build across 1.x point releases.

**What is green today, without DuckDB:** the engine runs the query through the
real operator tree and is diffed against an **independent reference oracle**
(`oracle/reference_oracle.*`) that computes the result monolithically over the
whole table — sharing **no** code path with the operators (it does reuse the
frozen, separately-oracle-bound `expr::evaluate`, which is correct: WP-3's new
surface is the *operator assembly*, not expression semantics). This makes the M1
harness, comparator, generators, and the mutation self-test fully runnable and
green now; DuckDB plugs into the same seams as the authoritative cross-check once
staged. **Honesty note:** I cannot claim a byte-for-byte-vs-DuckDB green until the
amalgamation is staged; every green below is engine-vs-reference.

## 6. Discovered latent issue (note to reviewer; ICR candidate)

ASan/assert caught a real latent issue while standing up the empty-table edge:
**`core::compact_column` (and therefore `expr::evaluate`) aborts on a
`row_count == 0` batch** — for an empty result both the new and source data
pointers are `nullptr`, tripping `compact_column`'s `out.data() != in.data`
("must be out-of-place") assert. WP-1/WP-2 never exercised length 0 (their
generators start at length 1). 

Scope/impact: the **WP-3 engine pipeline is unaffected** — Scan returns `nullopt`
for a 0-row table and Filter skips empty batches, so the operators never call
`evaluate` on a 0-row batch. The crash only appeared in my **reference oracle**,
which evaluates over the whole-table batch directly. I fixed it **in my own code**
(no frozen edits): the reference oracle short-circuits `n == 0`, and Filter/Project
defensively guard `row_count == 0` (Project emits a typed empty batch; Filter
pulls again). I did **not** touch the frozen `core`/`expr`.

**Recommendation (for the final review, not actioned here):** consider an ICR to
make `compact_column` total at `n == 0` (treat empty as out-of-place), and have
WP-9 add a 0-row case to the generator catalog. Until then, the convention
"operators never emit empty batches" (now in the frozen `operator.h` doc) keeps
this from biting. This is a candidate entry for the WP-15 pinned-bug story.

## 7. Mutation self-test #1 (the oracle bites)

`tests/oracle_mutation_test.cpp`, two test-only planted mutants (never linked into
the engine), each run through the engine pipeline and diffed against the
oracle(s):
- **`BuggyFilter`** — inverts the pass decision (keeps the rejected rows, incl.
  NULL-predicate rows). Caught: `engine=110 oracle=90`.
- **`BuggyScan`** — drops the final partial batch (the SIMD-tail/last-batch
  omission hotspot, §12). Caught: `engine=83 oracle=90`.

Each test also asserts the **correct** operator passes the same diff, so the check
is shown to both fail-on-bug and pass-on-correct. When DuckDB is staged the same
mutants are additionally diffed against DuckDB.

## 8. Assumptions

- Batch size is a multiple of 64 (zero-copy validity sub-views). Documented and
  enforced; tests use 64/256/2048.
- TS compared as raw int64 ns (`BIGINT` in SQL). No temporal semantics in WP-3.
- First-cut Filter compacts (D6); selection-vector *output* flow is a later
  measured optimization (the format already carries it and Filter consumes an
  input selection vector correctly).
- F64 epsilon 1e-9/1e-9 is for element-wise ops; aggregate epsilons are WP-5.

## 9. ICRs

None taken against frozen interfaces. One **ICR candidate flagged** for the
reviewer (§6: `compact_column` on `n == 0`); handled locally for WP-3 without
editing frozen code, so no interface change is requested at this time.

---

## 10. Exact commands

All run from the repo root. Host: `host=mac-m*`, `isa=neon` (preliminary; this WP
makes no perf claims — correctness only).

```bash
# (A) Full repo gate: forbidden-include gate + bite self-test, then
#     configure+build+ctest on release / asan(UBSan) / tsan, then Highway smoke.
scripts/ci.sh                       # => "ci.sh: ALL GATES GREEN"

# (B) Per-preset build + the WP-3 tests on their own.
cmake --preset release && cmake --build --preset release -j
ctest --preset release -R 'ops_test|oracle_differential_test|oracle_mutation_test' --output-on-failure
cmake --preset asan    && cmake --build --preset asan    -j
ctest --preset asan    -R 'ops_test|oracle_'   --output-on-failure   # ASan/UBSan
cmake --preset tsan    && cmake --build --preset tsan    -j
ctest --preset tsan    -R 'ops_test|oracle_'   --output-on-failure

# (C) The M1 proof: scan->filter->project diffs green vs the oracle.
./build/oracle_differential_test --seed 20260614     # CI seed (QE_CI_SEED)

# (D) Mutation self-test — SHOW the differential catching the corrupted operators.
./build/oracle_mutation_test -s 2>&1 | grep -i caught
#   => inverted-filter mutant caught: row count mismatch: engine=110 oracle=90
#   => dropped-tail   mutant caught: row count mismatch: engine=83  oracle=90

# (E) Replay any generated case from its seed (deterministic; seed is printed).
./build/oracle_differential_test --seed 12345        # same input every run

# (F) Fuzz: fresh random seed each run (prints the replay command on start).
./build/oracle_differential_test

# (G) After staging DuckDB (third_party/duckdb/, see VENDORING.md): the same
#     tests additionally diff against DuckDB — no test code changes.
cmake --preset release           # prints "DuckDB backend ENABLED"
cmake --build --preset release -j
ctest --preset release -R 'oracle_' --output-on-failure
```

**Result on this host (mac, neon):** `scripts/ci.sh` = ALL GATES GREEN; 16/16
ctests pass under release, asan/UBSan, and tsan. `oracle_differential_test`
exercises 300 random scan→filter→project queries (schema/data/query) per run plus
explicit empty-table / all-pass / all-fail / null edge cases, across batch sizes
{64,256,2048}; 5 fresh fuzz seeds + the pinned CI seed all green; same-seed runs
reproduce identically (312 assertions). The mutation self-test catches both
planted mutants and passes the correct operators.
