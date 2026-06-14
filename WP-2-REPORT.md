# WP-2 Report — Expression evaluation

Status: **complete, all gates green (re-submission after audit bounce).**
`scripts/ci.sh` is green end-to-end: from-scratch gate (clean on
`core simd expr ops plan tsx`) + its bite self-test, then release + ASan/UBSan +
TSan each **13/13 tests** with **committed fixed seeds** (deterministic — verified
**150 consecutive green runs**, 50 per preset, plus 2 full `ci.sh` runs), then
Highway smoke (dispatched target NEON on this host). This WP turns `expr/` from an
empty skeleton into the frozen expression engine WP-3 (project/filter) and WP-8
(plan) build on.

> **Audit bounce resolved (see §9 for the full pinned-bug writeup).** A real bug
> (vectorized `F64→I64`/`TS` cast corrupting large-magnitude integral doubles) and
> a rigor gap (CI ran off a fresh random seed each run) were found. Both are
> fixed: kernel corrected, a permanent value-level **and** seed-level regression
> added, and CI made deterministic with committed seeds. Note: the audit guessed
> the culprit was integer `Mul`/`Div`/`Mod`; node-level localization showed it was
> actually the `F64→I64` **cast rounding** (also an I64-typed, data-dependent,
> SIMD-path-vs-scalar divergence — same §12 family).

Host of record for the runs below: `host=apple-silicon isa=NEON` (Apple clang,
macOS). Per RIGOR.md §host-honesty these are **relative/correctness** runs only
— no perf numbers are claimed in this WP (WP-2's gates are correctness, not speed).

---

## 1. The public API proposed to freeze (`expr/expr.h`)

Minimal surface; everything else (`kernels.h`, `eval_internal.h`, the `*_kernels`
TUs) is internal and may evolve.

**Operator vocabularies:** `ArithOp{Add,Sub,Mul,Div,Mod}`,
`CmpOp{Lt,Le,Gt,Ge,Eq,Ne}`, `LogicOp{And,Or,Not}`.

**`struct Scalar`** — one typed literal (`Type`, `is_null`, `int64 i`,
`double f`) with factories `i32/i64/f64/boolean/ts/null(Type)`.

**`class Expr`** — value-semantics handle to an immutable shared `Node` tree;
`Expr::type()` is the inferred result type. (`Node`/`NodeKind` are public so the
evaluator can switch on them, but callers build trees only through the factories.)

**Builders** (all infer types + validate, throwing `std::invalid_argument` on a
type error; numeric binary ops auto-insert promotion casts):
- `col(Type, index)` and `col(const Schema&, name)`
- `lit(Scalar)`
- `arith(op,a,b)` + `add/sub/mul/div/mod`
- `cmp(op,a,b)` + `lt/le/gt/ge/eq/ne`
- `logic_and/logic_or/logic_not`
- `cast(e, Type)`

**Evaluation:**
```cpp
enum class Backend { Vector, Scalar };
OwnedColumn evaluate(const Expr&, const Batch&, Backend = Backend::Vector);
```
`evaluate` produces a **dense** `OwnedColumn` of `batch.row_count` values: the
batch's selection vector is applied while materializing `Column` leaves (via WP-1
`compact_column`), so every intermediate is dense and the op×type kernels always
see two same-length, same-type operands. The `Backend` param defaults to `Vector`
(WP-3/WP-8 ignore it); `Scalar` drives the independent reference twins and exists
so tests assert `scalar == vector` over whole trees.

**Design choices worth ratifying:**
- *Positional column refs are the contract; name resolution happens at build time*
  (`col(schema,name)` resolves to an index+type immediately). Eval is purely
  positional — no schema needed at eval time.
- *Numeric promotion is done by inserting `Cast` nodes at build time* (rank
  `F64>I64>I32`). This keeps the kernel matrix **monomorphic** — no kernel does
  mixed-type arithmetic. TS and BOOL are **not** numeric: arithmetic on them
  throws; cast them explicitly first (TS is int64-ns, so `cast(ts, I64)` is a
  reinterpret).
- *Results are always dense.* Filter (WP-3) may instead want a selection vector;
  that's WP-3's concern — expression eval returns a materialized column. (Possible
  future ICR if WP-3 wants a fused predicate-to-selection path; not needed to
  freeze this surface.)

---

## 2. Null-propagation truth table (implemented in `eval.cpp`, tested exhaustively)

Tested in `tests/expr_null_truthtable_test.cpp` under **both** backends.

**Arithmetic (`+ - * / %`) and comparison (`< <= > >= == !=`): null-if-any.**
Result is NULL iff any operand is NULL. Implemented by `propagate_nulls_and`
(out validity = a.valid AND b.valid, reusing WP-1 `simd::bit_and_vec`; stays on
the all-valid fast path iff both inputs are all-valid).

**Division / modulo by zero (integer AND float): NULL.** Matches DuckDB's default
and keeps integer division UB-free. Implemented as an extra validity pass
(`null_where_divisor_zero`) ANDed into the result.

**Logical AND/OR/NOT — SQL/Kleene three-valued** (N = NULL):

| AND | F | N | T |     | OR | F | N | T |     | NOT |   |
|-----|---|---|---|-----|----|---|---|---|-----|-----|---|
| F   | F | F | F |     | F  | F | N | T |     | F   | T |
| N   | F | N | N |     | N  | N | N | T |     | N   | N |
| T   | F | N | T |     | T  | T | T | T |     | T   | F |

Note `F∧N=F` and `T∨N=T` — a known value dominates a NULL; a plain "null-if-any"
rule is **wrong** here (and is planted mutant #2). Implemented via a **tri-state**
encoding (0=FALSE, 1=NULL, 2=TRUE) that makes Kleene `AND=min`, `OR=max`,
`NOT=2−x`; eval converts a BOOL column's (value bytes + validity bitmap) to/from
this encoding around the kernel.

**Cast:** NULL propagates (input NULL ⇒ output NULL). A value out of the target
integer range, or a non-finite float cast to an integer, ⇒ NULL
(`null_where_cast_oor`). `float→int` rounds **half away from zero** (DuckDB
semantics). The out-of-range/round predicate (`expr/cast_round.h`) is the **single
source of truth** shared by both kernel twins and eval, so the produced value and
the NULL decision cannot drift.

---

## 3. Arithmetic overflow policy (documented; UB-free, UBSan-clean)

- **signed int `+ - *`** → two's-complement **wraparound**, computed through
  unsigned intermediates (well-defined in C++20). The SIMD path uses Highway's
  non-saturating ops (wrap at hardware level); the scalar tail uses the unsigned
  cast. No C++ signed-overflow UB executes — confirmed by 2000+ random trees
  under UBSan (5 seeds × 400 iters) plus the `INT_MIN`-seeded kernel tests.
- **signed int `/ %`** → divisor `0` ⇒ NULL (never divides by zero); the
  `INT_MIN / -1` (and `INT_MIN % -1`) overflow is computed via wrapping negation
  (⇒ `INT_MIN`, and `0`), never UB.
- **float** → IEEE-754 (`±inf`/`NaN`), except `/0` and `%0` which map to NULL.

This is the policy WP-3 diffs against DuckDB: **integer = exact equality (D11)**,
float aggregates would use D11 tolerance (not relevant to elementwise expr kernels
— scalar and vector are bit-identical elementwise, so this WP compares floats
exactly, with NaN treated as equal-to-NaN).

---

## 4. How the matrix is templated while keeping the scalar twin independent (D17)

Each family is two TUs with identical signatures (`*_vec` / `*_scalar`):

| family | vector path (Highway) | independent scalar twin |
|---|---|---|
| arith | `arith_kernels.cpp` — `template<T>` per op; `if constexpr` splits float/int; Highway `Add/Sub/Mul/Div` | `arith_scalar.cpp` — `template<T>` plain loops; unsigned-wrap helpers |
| compare | `compare_kernels.cpp` — Highway `Lt/Le/...` mask → 1/0 lane → byte | `compare_scalar.cpp` — plain `a<b` loops |
| logic | `logic_kernels.cpp` — `Min/Max/(2−x)` on tri-state bytes | `logic_scalar.cpp` — **explicit truth-table branches** (deliberately not min/max) |
| cast | `cast_kernels.cpp` — Highway promote/demote/convert + vectorized round-half-away | `cast_scalar.cpp` — `std::round` + per-element convert |

The vector and scalar paths are genuinely different code (e.g. logic: SIMD min/max
vs. branch tables; arith int-add: Highway wrapping add vs. unsigned-cast add), so
`scalar == vector` compares two implementations, not one instantiation — exactly
the D17 requirement.

**Documented scalar corners (honest, mirroring WP-1's `gather8_scalar` /
int-division decisions):**
- Integer `/` and `%` have **no portable SIMD instruction** (none on NEON/AVX), so
  their "vector" path runs an independently-written scalar guarded loop. Float
  `/` is vectorized; float `%` (`fmod`) is scalar.
- **Cast to/from BOOL** (1-byte lanes vs wide numeric lanes) has no clean uniform
  SIMD form, so it uses an independent per-lane loop inside the vector TU. All
  numeric↔numeric casts (I32/I64/TS/F64) **are** vectorized.

---

## 5. Tests / rigor (the WP-2 gates)

All under `tests/`, reusing `tests/wp1_test_main.cpp` (prints the seed; `--seed N`
replays). **CI runs every randomized test with a committed fixed seed**
(`QE_CI_SEED=20260614` in `CMakeLists.txt`) so green is reproducible, not luck
(D13). **Discovery/fuzz mode** is still available — run any test binary with **no
`--seed`** and it picks+prints a fresh seed each run (loop it to hunt). Counts are
stable across release/asan/tsan.

- **`expr_scalar_vector_test`** — `scalar == vector` for **every op×type kernel**
  over seeded random inputs at **boundary lengths** (`0,1,2,7,…,63,64,65,…,2047,
  2048` — the SIMD tail). Includes `0`/`-1` divisors and `INT_MIN` to exercise the
  guarded integer paths.
- **`expr_null_truthtable_test`** — the §2 truth table exhaustively (all 9 Kleene
  rows for AND/OR, all 3 for NOT; null-if-any for arith/compare; div-by-zero NULL;
  cast NULL + out-of-range), both backends.
- **`expr_property_test`** — seeded fuzz over **random expression trees** (depth 3,
  400 trees/run) across random batches **with nulls and selection vectors**, at
  boundary lengths; asserts `scalar == vector` on **values AND validity**. Stressed
  5× with random seeds under UBSan — all green.
- **`expr_mutation_test`** — the **mutation self-test**, three planted bugs, one
  per §12 expression hotspot, each shown caught by the exact check the real suite
  uses:
  1. **SIMD tail dropped** in a comparison kernel → caught by `scalar==vector` at
     length 65.
  2. **Three-valued logic degraded to null-if-any** (`F∧N` wrongly NULL) → caught
     by the AND truth table.
  3. **All-valid fast path dropping a real null** → caught by null propagation.
- **`expr_property_pinned_castbug`** (ctest entry) — replays the exact failing
  tree (`expr_property_test --seed 14745959599513406255`) that first exposed the
  pinned bug (§9). Plus a direct value-level regression inside
  `expr_scalar_vector_test` ("REGRESSION (pinned bug): …"). Both fail before the
  kernel fix and pass after.

I verified the suite **bites** (not just that it passes): holding the planted
mutant to the real-kernel differential turns the run RED (see command in §6),
and the real kernels pass it green.

---

## 6. Exact commands

All from repo root. Randomized tests print their seed; replay with `--seed N`.

**Full gate (from-scratch + release/asan/tsan + Highway smoke):**
```
scripts/ci.sh
```

**Per preset (configure + build + ctest):**
```
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    -j && ctest --preset asan
cmake --preset tsan    && cmake --build --preset tsan    -j && ctest --preset tsan
```

**Just the WP-2 tests (any preset's build dir):**
```
ctest --preset release -R expr --output-on-failure
```

**scalar == vector and the null truth table directly:**
```
./build/expr_scalar_vector_test
./build/expr_null_truthtable_test
```

**Property/fuzz (discovery mode — fresh seed each run) + deterministic replay:**
```
./build/expr_property_test                       # discovery: new seed printed
for i in $(seq 1 50); do ./build/expr_property_test || break; done   # loop-fuzz
./build/expr_property_test --seed 2211200335224136098                # replay
```

**Pinned cast bug — the regression (deterministic):**
```
ctest --preset release -R expr_property_pinned_castbug --output-on-failure
./build/expr_property_test --seed 14745959599513406255   # the failing tree, now green
./build/expr_scalar_vector_test --seed 1                 # value-level regression
```

**Determinism proof (CI uses fixed seeds — repeat any number of times):**
```
for i in $(seq 1 50); do ctest --preset release -R expr || { echo "RUN $i RED"; break; }; done
# (verified 50× each on release / asan / tsan = 150 consecutive green)
```

**Mutation self-test (catch demonstrated):**
```
./build/expr_mutation_test
```

**Show the mutation suite going RED on the mutant, then green again** (proves the
checker can fail — temporarily holds the planted mutant to the differential a real
kernel must pass):
```
cp tests/expr_mutation_test.cpp /tmp/emt.bak
perl -0pi -e 's/CHECK\(mut != ref\);    \/\/ the MUTANT is caught: its dropped tail diverges/CHECK(mut == ref);/' tests/expr_mutation_test.cpp
cmake --build --preset release --target expr_mutation_test -j && ./build/expr_mutation_test   # 1 failed
cp /tmp/emt.bak tests/expr_mutation_test.cpp   # restore (file is untracked pre-commit)
touch tests/expr_mutation_test.cpp
cmake --build --preset release --target expr_mutation_test -j && ./build/expr_mutation_test   # 13/13 green
```
(Observed: `assertions: 13 | 12 passed | 1 failed` on the mutant; `13 passed`
after restore.)

---

## 7. Assumptions / notes for the final review (and WP-3)

1. **`float→int` cast rounding = round-half-away-from-zero** (`std::round`), and
   **out-of-range / non-finite int casts ⇒ NULL** (DuckDB raises; we model the
   failure as NULL). These are the two cast semantics most likely to need a
   tweak when WP-3 actually diffs casts against DuckDB. They are isolated in
   `expr/cast_round.h` (one predicate, one rounding fn) so a change is one edit,
   not a kernel rewrite. **Flagging for WP-3 to confirm against DuckDB.**
2. **Division/modulo by zero ⇒ NULL** for both integer and float — assumed to
   match DuckDB's default. Also flagged for WP-3 confirmation; isolated in
   `eval.cpp::null_where_divisor_zero`.
3. **TS arithmetic is disallowed** (cast to I64 first). DuckDB timestamp
   arithmetic has interval semantics we deliberately don't model in the numeric
   core (consistent with D7/non-goals). Comparison `TS<TS` is allowed (int64).
4. **No ICRs.** The frozen `core/`+`simd/` signatures were sufficient — I built
   entirely on `OwnedColumn`/`OwnedBatch`/`compact_column`, the validity bitmap
   primitives, and `simd::bit_and_vec`. Nothing in core/simd was edited.

## 8. Pinned bug (rigor item 10 / §12) — writeup for WP-15

**One-line:** the vectorized `float→int` round-half-away-from-zero corrupted
already-integral doubles with `|x| ≥ 2^52`, so `CAST(F64 AS BIGINT/TIMESTAMP)`
produced values off by one versus the scalar twin and versus DuckDB.

**Symptom / how it was caught.** `expr_property_test` (random expression trees,
scalar==vector on values *and* validity) diverged: `--seed 14745959599513406255`,
iter 199, `n=2047`, result type I64. It reproduced on TSan with a different seed
and at `n=63` — all I64, all at non-multiple-of-vector-width lengths, which made
it *look* like a classic SIMD tail/remainder bug in integer `Mul`/`Div`/`Mod`.

**Root cause (the interesting part).** It was **not** the tail and **not**
arithmetic. Node-by-node localization (evaluate every subtree under both backends,
find the deepest diverging node) pinned it to a `Cast F64→I64` node. The kernel
rounded with the common trick:
```cpp
r = Trunc(x + copysign(0.5, x));   // WRONG for large |x|
```
For `|x| ≥ 2^52` a double's ULP is ≥ 1, so `x ± 0.5` is **not representable** and
rounds back — for an already-integral `x` it rounds to the *next* integer. Example
from the failing run: `x = -7901924385117227.0` (exactly integral) →
`x − 0.5` rounds to `-7901924385117228.0` → `Trunc` → `-7901924385117228` (off by
one). The scalar twin used `std::round`, which is correct, so `scalar == vector`
flagged it.

Why it looked length-correlated: the value only has to *appear* in a batch to
trigger; which random values land at which positions depends on `n` (different `n`
⇒ different RNG draws), so the failure tracked particular lengths by coincidence,
not because of the partial final vector. Why only I64/TS: for `F64→I32` such
magnitudes are out of int32 range, so they are NULL'd and the corrupt value is
masked — the bug was invisible there.

**The fix** (`expr/cast_kernels.cpp`, `RoundAway`): truncate toward zero, then add
`±1` only when the **exact** fractional part is `≥ 0.5`:
```cpp
t = Trunc(x); frac = x - t;                       // both exact
bump = (|frac| >= 0.5) ? copysign(1.0, x) : 0;
r = t + bump;
```
For `|x| ≥ 2^52`, `frac == 0` ⇒ `r == x` (correct). For `|x| < 2^52`, `t`, `frac`,
and `t ± 1` are all exactly representable, so it matches `std::round` bit-for-bit.

**The regression (cannot silently return).**
1. *Value-level*: `expr_scalar_vector_test` → "REGRESSION (pinned bug)…" casts the
   exact failing value, `±2^52`, `2^53−1`, halfway cases, and a 5000-sample sweep
   of large integral doubles to I64 and TS; asserts `cast_vec == cast_scalar` and
   that integral inputs map to themselves.
2. *Seed-level*: ctest `expr_property_pinned_castbug` replays
   `expr_property_test --seed 14745959599513406255`.
Both were demonstrated RED on the buggy kernel and GREEN after the fix.

**Lesson for the catalog (WP-9/WP-15):** "rounded with `x + 0.5`" is a recurring
SIMD-cast defect; the scalar/vector differential over a wide value range
(including `|x| ≥ 2^52`) is what catches it. The mutation catalog should carry a
"float→int round via +0.5" entry.

## 9. Definition-of-done checklist
- [x] Conforms to a minimal frozen interface (`expr/expr.h`).
- [x] From-scratch (forbidden-include grep clean, incl. `expr/`).
- [x] Scalar twin present and **independent** for every vectorized kernel.
- [x] Null-propagation truth table documented + exhaustively tested.
- [x] Mutation self-test: ≥1 (delivered **3**, one per hotspot) shown to bite.
- [x] ASan/UBSan green (re-run); TSan green.
- [x] Seeds printed & replayable (`--seed N`); **CI uses committed fixed seeds**
      (deterministic — 150 consecutive green runs verified).
- [x] **Pinned bug fixed** with a permanent regression that fails-before /
      passes-after (§8), wired into ctest.
- [x] Every claim here has a reproduce command (§6); no perf numbers claimed.
- [x] Reproducibility scripts (`ci.sh`, presets) still green after the change.
