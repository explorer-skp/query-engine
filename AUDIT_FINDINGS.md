# Full-codebase verification audit — 2026-07-05

Independent audit of the entire engine, oracle, test, bench, and script layers,
performed by eight parallel deep-review passes (one per subsystem, each reading
its module line-by-line against RIGOR.md and reviewer_brief.md) plus a
ground-truth CI rerun. Every finding below was verified against the actual
source (quoted file:line), and the highest-severity ones were independently
re-confirmed. DuckDB-semantics claims were verified empirically by compiling
probes against the staged amalgamation, and the Gorilla codec was fuzzed
standalone under ASan/UBSan (1.2M adversarial values, bit-exact).

## Ground truth: CI rerun (this machine, host=mac, isa=neon)

`./scripts/ci.sh` — forbidden-include gate + bite self-test: **green**.
Release preset: **51/51 green**. ASan/UBSan preset: **51/51 green**.
TSan preset: **50/51 — RED**:

```
47 - validity_gate_test  ***Failed
tests/validity_gate_test.cpp:122: CHECK( spread > cfg.max_probe_spread ) is NOT correct!
  values: CHECK( 1.35113 > 2 )
```

The oversubscription self-test expects a deliberately-loaded machine to show a
>2.0 timing spread; under TSan's uniform ~5-15x slowdown the *relative* spread
collapses (1.35). This is an environment-sensitive rigor-layer test, not an
engine bug — but per rule 6 ("Sanitizers are a gate... Re-run them; don't
assume"), the TSan gate is currently red on this machine, and ci.sh aborted
before its step [3/3]. Note this rerun was the FIRST time the two WP-7b string
tests ran under TSan at all (the old build-tsan/ predated them — it had 49
registered tests, not 51); both passed.

Overall verdict: **the tested surface is genuinely solid** — the differential
oracle, mutation catalog, scalar/vector twins, radix sort, join machinery,
compression codecs, and the pinned cast-bug fix all check out under adversarial
line-level review. The real defects cluster in exactly the places the test
generators never reach: **the parallel layer x strings, NaN semantics, and the
oracle's own error handling**. That is the expected failure mode of a
generator-driven oracle, and it is fixable.

---

## CRITICAL — real bugs producing crashes or silent wrong results

### C1. Parallel exec layer is broken for VARCHAR (three independent bugs)
Any `supported()` plan touching a STR column through `exec::ParallelEngine`
is wrong or crashes, at every thread count including 1. `supported()`
(exec/parallel.cpp:326-340) gates on plan *shape* only — STR sails through.
The WP-10b differential can never see this: `rand_type`
(oracle/generators.cpp:151-159) never emits STR.

- **C1a — `MorselScan` never sets `Column::dict`** (exec/morsel_scan.cpp:48-63).
  `Column view;` leaves `dict` indeterminate (`Column` is a plain aggregate,
  core/column.h:56-68). The frozen Scan sets it explicitly with a warning
  comment about exactly this (ops/scan.cpp:42-46); MorselScan's header comment
  claims its view construction is "intentionally identical to ops/scan.cpp" —
  false by the one load-bearing line. Every STR consumer then dereferences a
  wild pointer (oracle/result_set.cpp:41, ops/aggregate.cpp:170, sort, join).
- **C1b — `key_of` encodes STR group keys from `Cell.i`, which is always 0**
  (exec/parallel.cpp:250-271). All distinct string groups collapse into one
  during partial-aggregate merge → silently one row with the sum of everything.
- **C1c — MIN/MAX(STR) partials folded as integers** (exec/parallel.cpp:218-232,
  finalize at :539-547): result is empty strings, never `Cell.s`.

Fix: `view.dict = oc.dict();` (+ value-init `Column view{};`), then either
implement STR in key_of/fold/finalize or make `supported()` reject STR plans —
and add STR coverage to the parallel differential plus a mutation self-test
(rule 4: right now this checker cannot fail).

### C2. NaN handling in F64 aggregation/sort diverges from DuckDB and is ISA-dependent
DuckDB orders NaN greater than everything. The engine has no NaN policy, and
the oracle generators never emit NaN (generators.h:16-17), so the entire class
is invisible to the differential:

- **Grouped MIN/MAX drops NaN** (ops/aggregate.cpp:137-149 — `std::min/max`
  with NaN returns the non-NaN arg) → group `{NaN}` emits **±inf** (the init
  identity) where DuckDB emits NaN.
- **Global scalar twin is position-dependent** (ops/agg_scalar.cpp:38-50):
  NaN at index 0 → NaN; NaN elsewhere → ignored.
- **Global vector path is ISA-divergent**: `hn::Min` on x86 is `_mm_min_pd`
  (returns 2nd operand on unordered) but NEON `vminq_f64` returns NaN — the
  same input gives different answers on ARM vs x86, violating the two-ISA rule;
  scalar==vector passes only because tests never feed NaN.
- **Window tumbling** (tsx/window.cpp:133-143) has the same fold → all-NaN
  bucket emits +inf, while the **sliding** deque path (tsx/window.cpp:452-501)
  is order-dependent (`{3.0, NaN}` → NaN but `{NaN, 3.0}` → 3.0) — the two
  modes of one operator disagree with each other.
- **Sort comparison path**: NaN breaks strict weak ordering
  (ops/sort.cpp:78-81) → `std::stable_sort` UB. Documented scope cut
  (sort.h:40-42) but unguarded.

Fix: adopt DuckDB's ordering (NaN greatest) in one shared decision, apply in
scatter/kernels/twins/sort comparator, add NaN to generators + a NaN mutation
self-test. (Hash keys already canonicalize NaN correctly — hash_internal.cpp:43;
it's the *value* path that was never specified.)

### C3. The oracle silently degrades: every `DuckDBError` is swallowed, and the fallback reference is circular on expression semantics
All 16 catch sites (tests/catalog_check_util.h:64-67; the `catch
(DuckDBError&) { /* ignore */ }` in agg/join/sort/plan/asof/window/compress/
oracle/wp7b mutation tests; wp10b sets `have_duck=false` for the whole run)
convert a DuckDB raise into a non-failing skip (`MESSAGE` never fails doctest).
No test anywhere REQUIREs that DuckDB actually executed when staged. Therefore
a regression in oracle/sql_render.cpp / plan_sql.cpp that produces *invalid*
SQL disables the entire DuckDB layer with CI green — while MUTATION_CATALOG.md
claims verdicts are "decided against DuckDB". Meanwhile the reference oracle
computes filter/project semantics via the engine's own `expr::evaluate`
(oracle/reference_oracle.cpp:252,466,481 — documented, but load-bearing here).
Demonstrated latent trigger: SUM/AVG(BOOL) — engine + reference accept it,
DuckDB binder errors (verified vs the staged amalgamation), so such plans are
checked engine-vs-itself-adjacent forever.

Fix (5 lines of leverage): a smoke test `REQUIRE_NOTHROW(run_duckdb(...))` when
`duckdb_available()`, and make generated-case harnesses treat `DuckDBError` as
failure (or count skips and CHECK(skips==0)), keeping leniency only for
deliberately-divergent hand cases.

---

## HIGH — correctness-risks (latent: real bugs one contract-legal input away)

- **H1. Reference sliding-window sorts NULL timestamps as t=0**
  (oracle/reference_oracle.cpp:932-938): engine sorts NULLS LAST
  (tsx/window.cpp:63-70), DuckDB defaults NULLS LAST (verified) — the reference
  is the odd one out. Masked only because gen_window_case never emits NULL ts.
- **H2. Sliding `OVER (ORDER BY …)` render omits `ASC NULLS LAST`**
  (oracle/plan_sql.cpp:299-300), relying on DuckDB's session-configurable
  default — violates the project's own "never rely on a backend default" rule,
  which order_by_sql itself honors (sql_render.cpp:157-158).
- **H3. Selection-vector physical-walk family**: `canonicalize_str_key`
  (ops/aggregate.cpp:162-175) and the join's STR canonicalization
  (ops/join.cpp:52-58) walk ALL physical rows including sel-excluded ones, and
  `StringDict::at` is unchecked (core/string_dict.h:54-59) → OOB read the day
  any producer emits sel-batches with undefined dead slots (filter.h names
  sel-flow as a planned optimization). Same class: `cmp_str`
  (expr/eval.cpp:191-192) resolves dictionary codes of NULL lanes before
  null-masking — NULL STR lanes hold indeterminate codes (test builders write
  none), so this is UB that current tests dodge by accident. Fix: iterate via
  `sel_at`, skip invalid lanes, and bounds-check `StringDict::at` in debug.
- **H4. As-of tolerance check has signed-overflow UB** (tsx/asof.cpp:272):
  `tv.t - build_ts[cand]` overflows for spans > INT64_MAX (probe t=INT64_MAX,
  build t=-2). Grammar keeps ts in ±25k so the differential can't fire it.
  One-line fix with uint64 modular arithmetic (WP-14 does this correctly).
- **H5. Release builds compile out guards that docs call "guarded errors"**:
  group-id exhaustion assert (ops/hashtable.cpp:60 — at 2^32 groups, kNoGroup
  is written as a real group id, corrupting the table; hashtable.h:60-63
  promises "guarded error, not silent wraparound"); join key-type check is
  `#ifndef NDEBUG` only (ops/join.cpp:104-109); window sliding STR partition
  key is assert-only (tsx/window.cpp:169-171 — with NDEBUG all STR partitions
  silently merge); asof F64/BOOL-timestamp guard likewise (tsx/asof.cpp:76-80).
- **H6. `Buffer` accepts allocation failure silently** (core/buffer.cpp:14-23):
  on OOM `data_==nullptr` but `size_` stays set — breaks the class invariant,
  and the alignment assert passes for nullptr. Plus unchecked size multiplies
  (core/owned_batch.cpp:20,34: `len * byte_width` can wrap → small successful
  alloc, then OOB writes).
- **H7. Result-comparator weaknesses** (oracle/result_set.cpp): canonical sort
  is exact while cell compare is epsilon (:49-65 vs :78-82) — sub-epsilon
  float differences across the two sets can misalign rows (today prevented
  only by an unenforced "exact columns sort first" property of the
  generators); NaN returns 0 from cell_order → std::sort UB precisely when an
  engine bug produces a NaN; positional compare is engaged for any Sort-rooted
  plan (oracle/differential.cpp:49) with nothing enforcing total-order sort
  keys (all current callers append tiebreakers; the invariant lives in
  comments).
- **H8. Forbidden-include gate blind spots**
  (scripts/check_forbidden_includes.sh): the link-scan (:56-58) greps CMake
  files only under module dirs — but every target lives in the ROOT
  CMakeLists.txt, so `target_link_libraries(qe_ops PUBLIC duckdb_amalg)` would
  pass the gate; also case-sensitive (`#include <DuckDB.hpp>` bypasses, and
  macOS's FS would resolve it).
- **H9. Two ratio-reporting benchmarks have no validity gate**:
  bench_sort_main.cpp and bench_compress_main.cpp (the WP-7 and WP-14 headline
  ratios) emit host+isa-tagged JSON with no contamination check and
  unconditional exit 0 — contradicting the WP-10 "gate-checks before report"
  discipline that bench_ops/engine_vs_duckdb honor (`return gate.accepted ? 0 : 3`).
- **H10. `validity_gate_test` is environment-sensitive and now demonstrably
  red under TSan** (tests/validity_gate_test.cpp:122, spread 1.35 vs
  threshold 2.0 — see CI section). Related: the parallel mutation tests
  assume ≥2 workers actually win morsels with no synchronization
  (wp10b_parallel_mutation_test.cpp:115-123) — a scheduling-dependent flake
  risk in the opposite direction (mutant not flagged on a loaded machine).
- **H11. `gather32_vec` diverges from its scalar twin for indices ≥ 2^31**
  (simd/gather_kernels.cpp:31-32 — BitCast to int32 index lanes makes them
  negative; the 64-bit path is safe via PromoteTo). Unreachable at 2048-row
  batches; undocumented cap. `StringDict` also silently truncates past 4 GiB
  of interned bytes (uint32 offsets, core/string_dict.h:44-58).
- **H12. `HashTable::find()` is `const` but mutates shared scratch**
  (ops/hashtable.h:186-189) — a latent data race the day WP-10b's
  "shared-build future opt" lands. Needs at least a header warning.

---

## MEDIUM — docs/reports contradicted by code (rule-7/rigor hygiene)

- `tsx/asof.h:19` "NULL KEYS agree with DuckDB" — contradicted by
  WP-12-REPORT.md §3 and generators.cpp:530-541 (NULL keys are documented
  divergence #2, excluded from the DuckDB grammar).
- **No WP-13-REPORT.md exists** (WP-12/14 have theirs) — a definition-of-done
  gap; the deliberate O(n·P) float-recompute deviation from the spec's O(n)
  claim is recorded only in a code comment.
- MUTATION_CATALOG.md is stale: documents 13 entries, says "all 12 flagged";
  code has 17 (tests/mutation_catalog.cpp:41-93). Also the meta-test floor is
  6, so 8 of the 9 kOperatorOrchestration entries could be deleted without any
  test failing — the advertised "17" is enforced nowhere.
- `core/column.h:49-51` documents a debug assert for the invalid
  (all_valid=false, validity=nullptr) state that does not exist anywhere;
  `compact_column` silently promotes such a column to all-valid (real nulls
  dropped). WP-1-REPORT.md:53-55 repeats the false claim.
- exec/parallel.h:23-25 + WP-10b-REPORT.md claim the join build side is built
  "per worker"; it is rebuilt **per morsel** (exec/parallel.cpp:380 →
  full build lower+drain per morsel) — correct but far more redundant than
  claimed (~98 rebuilds for a 100k-row probe at 1024-row morsels).
  WP-10b-REPORT also declares the TSan gate but never states a TSan result.
- `expr/expr.h:28-30` float div-by-zero → NULL "matches DuckDB's default" —
  likely false for the pinned v1.1.3 (`ieee_floating_point_ops=true` → ±inf),
  and unverifiable by the oracle since division is excluded from the grammar.
  NaN comparison semantics (IEEE in engine vs DuckDB's NaN=NaN/greatest) are
  likewise undocumented as a divergence.
- `tests/expr_scalar_vector_test.cpp:44-46`: the "INT_MIN/-1" coverage claim is
  wrong — b[0] is always 0, so INT_MIN/-1 is never actually exercised (the
  guard itself is correct; a[2]=INT32_MIN would fix the test).
- `scripts/ci.sh:29-30`: step [3/3] runs `hwy_smoke --target-only` where the
  echo claims it checks `host_state` provenance — host_state is built but
  never executed by CI.
- `scripts/pick_rate.py:39-40`: thresholds contradict the docstring (p999<=20ms
  makes the documented p99<=50ms clause dead; likely swapped).
- `string_key_by_code` catalog entry's clean side runs the mutant's kNone copy,
  not the real HashJoin (tests/catalog_checks_strings.cpp:73-79) — mitigated by
  wp7b tests running the real path, but the catalog entry as registered
  validates a reimplementation.
- Stale comment: ops/sort_internal.h:34 still says compact_column "aborts" on
  n==0 (fixed long ago). tsx/window.h:13 calls truncation "floor division"
  (only true for t>=0; grammar-guarded). oracle/differential.h:11 references a
  nonexistent sql_oracle.h. Deque comment at tsx/window.cpp:496 says "keep
  oldest" but code keeps newest (behavior fine, comment wrong).

## LOW / robustness nits

- Unguarded varint decode on corrupt input (tsx/compress.cpp:82-92 — shift UB
  + overread past 10 continuation bytes; unreachable for self-encoded streams).
- Degenerate `max_load_factor < 1/capacity` → capacity doubles per key
  (ops/hashtable.cpp:46-49); `round_up_pow2` spins forever for >2^63 input.
- Worker exceptions in the parallel layer are std::terminate (no catch/rejoin,
  exec/parallel.cpp:377-448); `ParallelConfig.threads` uncapped.
- join_sql binds ON by column *name* (oracle/sql_render.cpp:243-246) though the
  engine contract is positional — duplicate names would mis-bind (test-layer
  only; generators use unique names). Unquoted identifiers in rendered SQL.
- uint32 row-index truncation unguarded (join_internal.h:86, sort.cpp:98).
- DuckDB-vs-reference backend choice is invisible in a green ctest log (only a
  configure-time message); GROUP BY renders rely on DuckDB's
  source-column-over-alias binder precedence (verified correct today);
  -0.0 literals load into DuckDB as +0.0 (cell_literal prints "-0").
- Project's defensive 0-row path emits STR columns without a dict
  (ops/project.cpp:35; unreachable today).
- `OwnedBatch::view()` mutates `mutable sel_view_` under const — data race if a
  batch is ever shared across threads (not currently done), and a second
  `set_selection` retargets previously-taken views.

## Verified sound (adversarially checked; no action needed)

- The WP-2 pinned bug fix (`RoundAway`, expr/cast_kernels.cpp:101-108) is
  mathematically correct for every double, including ±2^63 boundaries, NaN,
  ±inf, -0.0, ties, and the 2^52 regime; overflow-UB-free in both twins.
- Gorilla XOR encoder/decoder: line-audited AND independently fuzzed (1.2M
  adversarial values, streaming windows 1-17, ASan+UBSan) — bit-exact,
  including the lead-clamp path and the mean==64 edge. Delta-of-delta/zigzag/
  varint are fully modular-uint64, INT64_MIN-safe; streaming cursors resume
  correctly across arbitrary batch boundaries.
- Radix sort: per-pass stability, last-key-first LSD ordering, sign-flip for
  I32/I64/TS, DESC-via-complement (preserves tie order — NOT the reverse-output
  trap), absolute NULLS FIRST/LAST placement matching both the comparison path
  and the always-explicit SQL render.
- Join: multiplicity map, LEFT = probe-preserving on all three backends,
  batch-boundary fan-out carry-over (>2048 matches), null-padded build columns,
  composite-key null poisoning, empty-build LEFT path.
- Hash table: probe termination (lf≤0.95 guarantees an empty slot), growth
  preserves ids, null policies correct on both sides, vector/scalar probe
  produce identical group ids (shared row-sequential walk, bit-identical hash).
- Aggregate SQL semantics: COUNT(*) vs COUNT(col), all-null→NULL, empty-global
  →1 row, keyed-empty→0 rows, SUM(I32/I64)→I64 with CAST-to-BIGINT render.
- Oracle per-node engine-vs-SQL matrix: every currently-generated shape is
  consistent (filter NULL drop, full parenthesization, asof <= boundary +
  tolerance conjunct semantics, tumbling (t/W)*W ≡ t-t%W under truncation even
  for negative t, TS-as-BIGINT-ns end-to-end so no us/ns hazard, CompressedScan
  ctable/ctable_ref non-circular split).
- Test layer: zero dead tests (all 51 registered in all presets), comparator
  fails on row-count mismatch before any loop (no vacuous zero-row green),
  seeds printed + replayable everywhere, QE_CI_SEED wired, mutants genuinely
  single-defect and non-vacuously caught, sanitizer presets set real flags,
  -Werror real with justified carve-outs.

## Repo hygiene

`.preview/`, `DEPLOY.md`, `index.html` at the repo root are an unrelated
personal portfolio site (untracked). Move them out or .gitignore them so they
can't be committed here accidentally. (This audit file is also untracked —
keep, move, or delete as you see fit.)

## Suggested fix order (highest leverage first)

1. C3: DuckDB smoke test + fail-on-DuckDBError in generated-case harnesses.
2. C1: MorselScan dict line + STR gate/support in the parallel layer + STR
   parallel differential coverage.
3. C2: one NaN policy (DuckDB: NaN greatest) across agg/window/sort + NaN in
   generators + NaN mutation entry.
4. H1+H2: reference sliding NULL-ts sort key + explicit ASC NULLS LAST in OVER.
5. H3: sel_at iteration in the two STR canonicalizers + skip NULL lanes in
   cmp_str + debug bounds-check in StringDict::at.
6. H4 (one-line uint64 fix), H5 (throw instead of assert at the three sites),
   H6 (check Buffer alloc).
7. H8/H9/H10: root-CMakeLists link-scan, gate the two benches, make the
   oversubscription self-test TSan-aware (skip or scale threshold under TSan).
8. Doc sweep: asof.h NULL-keys line, MUTATION_CATALOG.md regen, column.h
   assert (implement it), WP-13 report, parallel.h per-morsel wording,
   ci.sh host_state step, pick_rate.py thresholds.

---

# FIX LOG — 2026-07-05 (applied in the suggested order)

67 files changed, ~1050 insertions / 145 deletions. Every fix follows the
project's own rule 4: where a fix closed a testing blind spot, the new checker
was PROVEN TO BITE (bug planted or fix reverted, red confirmed, restored, green
reconfirmed).

## C3 — oracle silent degradation: FIXED + trap proven
All 20 `catch (DuckDBError)` skip-sites across the differential/mutation
harnesses now FAIL loudly; the catalog helper propagates instead of swallowing
(tests/catalog_check_util.h). New anti-degradation smoke test in
oracle_differential_test proves DuckDB genuinely executes end-to-end when
staged, and WARNs loudly when a run is reference-only. Bite-proof: planted
`KREATE TABLE` in the loader → every case FATAL-ERRORs with "renderer/oracle
regression"; reverted → green.

## C1 — parallel × VARCHAR: FIXED + STR coverage added
- exec/morsel_scan.cpp: `Column view{}` + `view.dict = oc.dict()` (C1a).
- exec/parallel.cpp: `supported()` now refuses Aggregate plans with STR group
  keys or MIN/MAX(STR) inputs (`aggregate_exchange_speaks`) → correct
  single-thread fallback (C1b/C1c). Streaming/join/top-Sort STR plans stay
  parallel — they are value-level exchanges and correct once the dict rides.
- 3 new STR test cases in wp10b_parallel_differential_test (filter→project,
  join on VARCHAR keys across two dicts, and the gated-aggregate fallback).
  Bite-proof: dict line reverted → immediate SIGSEGV caught; restored → green.

## C2 — NaN policy: IMPLEMENTED engine-wide (DuckDB: NaN-greatest total order)
- Policy helpers in ops/agg_internal.h (`f64_less/min/max_total`); MIN identity
  is now NaN (all-NaN group ⇒ NaN, as DuckDB).
- Applied to: grouped scatter + global fold (ops/aggregate.cpp), scalar twins
  (ops/agg_scalar.cpp, independent isnan formulation), Highway kernels
  (ops/agg_kernels.cpp — NaN lanes masked before hn::Min/Max, so the reduction
  is ISA-DETERMINISTIC; the NEON-vs-AVX divergence is gone by construction),
  window tumbling fold + sliding deques (tsx/window.cpp), sort comparator
  (ops/sort.cpp via sort_internal.h `f64_cmp_total` — strict-weak-ordering UB
  gone), parallel partial merge (exec/parallel.cpp), all mutant faithful
  copies, the reference oracle (independent `ref_min/max_f64`), and the
  result-set canonical sort (NaN-total cell_order).
- Loader can now express NaN/±inf/-0.0 ('NaN'::DOUBLE etc. in duckdb_oracle) —
  which surfaced and fixed a REAL comparator bug: float_eq(-inf,-inf) was
  false (fabs(inf-inf)=NaN); equal-exact fast path added.
- Coverage: NaN scalar==vector kernel case; hand differential cases vs DuckDB
  in aggregate (grouped+global), sort (all directions × null orders), window
  (tumbling + sliding, three frame widths); catalog entry #18
  `agg_max_drops_nan` (mutant = pre-fix raw std::max) — meta-test proves it is
  flagged with clean passing.

## H1+H2 — oracle NULL-ordering: FIXED + coverage
Reference sliding sort key is now (valid, t, idx) NULLS LAST; OVER clause
renders explicit `ASC NULLS LAST`. New differential case (one NULL ts per
partition, deterministic across backends) — bite-proven by re-flipping the
reference to NULLS FIRST (red), restoring (green).

## H3 — selection-vector STR hardening: FIXED
Both STR canonicalizers (aggregate + join + the string_key_mutants faithful
copy) iterate live rows via sel_at, zero-fill dead slots, and no longer intern
filtered-out values; expr cmp_str skips invalid lanes (NULL lanes carry no
defined code); StringDict::at got the debug bounds assert.

## H4+H5+H6 — UB / release guards: FIXED
Asof tolerance span is modular uint64 (mutants mirrored); hashtable group-id
exhaustion throws in every build (doc promise now true); join key-type
mismatch throws; Window rejects STR keys/inputs and non-integer time columns;
AsofJoin rejects non-integer time columns and mismatched key types; Buffer
throws bad_alloc on allocation failure instead of breaking its invariant.

## H8+H9+H10 — gates: FIXED + self-tests extended
- check_forbidden_includes.sh: case-insensitive; new root-CMakeLists link scan
  that flags forbidden tokens on ENGINE targets only (no false positives on
  oracle/test DuckDB links); --self-test now also plants a root-CMake link and
  proves that branch bites.
- bench_sort + bench_compress: validity gate wired (verdict on stderr,
  `gate_accepted` in the JSON, exit 3 on rejection) — same rule as bench_ops.
- validity_gate_test oversubscription case: skipped LOUDLY under TSan (TSan's
  uniform slowdown flattens the probe spread — the exact CI red this audit's
  ground-truth run hit); release/ASan still run the real check.

## Docs — corrected
asof.h NULL-keys claim; expr.h float-/0 + NaN-comparison divergences spelled
out; generators.h NaN note; MUTATION_CATALOG.md regenerated (18 entries, was
documenting 13 with "all 12 flagged"); WP-13-REPORT.md written (was missing);
parallel.h/WP-10b-REPORT per-MORSEL build-side correction; column.h's promised
debug assert implemented in compact_column; ci.sh actually runs host_state and
greps its provenance; pick_rate.py thresholds unswapped (p99<=20ms,
p99.9<=50ms) and documented; stale comments (sort_internal abort note,
window.h "floor", deque "keep oldest", differential.h sql_oracle.h, dead
`ordered` var) cleaned.

## Verification status
Targeted suites green throughout (all differential tests incl. the new NaN /
NULL-ts / STR-parallel cases, all mutation tests, 18/18 catalog meta-test,
scalar==vector incl. NaN lanes, forbidden-include gate + extended self-test).
Full three-preset CI (release / ASan+UBSan / TSan), 2026-07-05 post-fix run:
`ci.sh: ALL GATES GREEN` — 51/51 on ALL THREE presets (the pre-fix baseline was
TSan 50/51 red on validity_gate_test; it now skips its timing probe loudly
under TSan and passes), forbidden-include gate + BOTH self-test branches
proven to bite, Highway smoke + host_state provenance verified. Reproduce:
`./scripts/ci.sh`.
