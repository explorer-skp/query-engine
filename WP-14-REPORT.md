# WP-14 Report — Time-series compression + compressed scan (Phase 2, M4)

**Scope:** Gorilla XOR float compression + delta-of-delta timestamps + zig-zag varint,
plus a **compressed scan** so decode feeds the engine end-to-end. Follows the WP-12
(as-of) pattern: a self-contained `tsx/` component + an **additive** plan node + oracle
wiring that renders the SAME plan to DuckDB SQL + a planted-mutant self-test + a catalog
entry. Effort: L. **Status: complete, all sanitizers green, diffs green vs the
independent reference AND DuckDB.**

---

## What shipped

New files (all under the from-scratch modules — grep-clean, no compression library):
- `tsx/compress.{h,cpp}` — the four codecs, `CompressedTable`, `CompressedScan`.
- `tsx/compress_kernels.{h,cpp}` + `tsx/compress_scalar.cpp` — the vectorized
  zig-zag-decode kernel (Highway) and its independently-written scalar twin.
- `tsx/compress_mutants.{h,cpp}` — the test-only planted-mutant decoder.
- `tests/compress_roundtrip_test.cpp`, `tests/compress_differential_test.cpp`,
  `tests/compress_scalar_vector_test.cpp`, `tests/compress_mutation_test.cpp`,
  `tests/catalog_checks_compress.cpp`.
- `bench/bench_compress_main.cpp` — the ratio/throughput reporting driver.

Additive edits (append-only; existing lines byte-unchanged):
- `plan/plan.h` / `plan/plan.cpp` — `CompressedScan` enumerator (after the prior last
  Phase-2 enumerator), `ctable` / `ctable_ref` `PlanNode` fields, `#include
  "tsx/compress.h"`, the `compressed_scan(ct, reference)` leaf builder, a `lower()` case,
  a `to_string()` case.
- `plan/plan_mutants.cpp` — a faithful `CompressedScan` lowering arm (no defect).
- `oracle/plan_sql.cpp` — `render_compressed_scan` + a `PlanKind::CompressedScan` case.
- `oracle/reference_oracle.cpp` — a `PlanKind::CompressedScan` arm in `eval_rs`.
- `tests/catalog_checks.h`, `tests/mutation_catalog.cpp`, `CMakeLists.txt` — registration.

---

## Design notes

### Codec framing (each column = a validity stream + a values stream)
Every column encodes to two independent byte streams: the **validity bitmap**, copied
**byte-for-byte** (`words(nrows)` 64-bit words, only when the column has nulls), and a
**values stream** holding **only the non-null values in row order**, per type:

- **F64 → Gorilla XOR.** First value raw (64 bits); each subsequent value XOR'd against
  the previous. A zero XOR is one bit; otherwise a leading-zero / meaningful-bit-run
  block (`reuse-window` control bit, or a new window with a 5-bit leading-zero count and
  6-bit meaningful length). Gorilla **never interprets the float**, so it is bit-exact
  across the entire double domain by construction — NaN payloads, ±0.0, ±inf, subnormals.
- **TS / I64 → delta-of-delta + zig-zag + varint.** The second difference of the series,
  computed in **modular uint64** (so it is well-defined and reversible for *every* int64
  input — no signed overflow, UBSan-clean), zig-zagged to a small unsigned magnitude,
  then LEB128 varint-coded.
- **I32 → zig-zag + varint** (sign-extended through the same uint64 zig-zag so decode
  reuses one kernel; delta intentionally omitted — kept simple).
- **BOOL → 1-bit-per-value bit-pack** of the non-null values.

The streams are **resumable**: a decoder cursor is just a byte position (varint) or bit
position (Gorilla/bool) plus the codec's running carry (prev value / prev delta, or the
Gorilla previous-value + window). So `CompressedScan` decodes the **next window** of each
column per `next()` — sequentially, **without re-decoding from row 0** — and `batch_size`
is free to vary. Because each batch materializes **fresh** `OwnedColumn`s with their own
freshly-built validity bitmap, there is **no** zero-copy word-alignment constraint (unlike
`ops/scan.h`): the round-trip test sweeps batch sizes `{1,7,13,64,256,2048}` to prove it.

### The lossless + nullable contract (and why null-slot bytes are exempt)
Stated precisely in `tsx/compress.h`: after `decode(encode(col))` the **validity bitmap
is preserved exactly** and **every non-null value is preserved bit-exactly** (compared via
the raw 8 bytes / bit pattern, never `==`, so NaN and −0.0 are checked correctly). The
bytes occupying a **NULL** slot are **semantically undefined and are not preserved**: a
null carries no value, the bitmap already records its nullness, and `Column` equality
respects validity, so a null slot's data bytes are never read. Encoding only the non-null
values + the bitmap is therefore lossless in the only observable sense — and it is what
makes the compression honest (a column of all-nulls compresses to just its bitmap; the
round-trip test asserts `ct.encoded(0).values.empty()` for that case). Decode zeroes each
fresh data buffer so a null slot is deterministic and never trips a reader of
uninitialized memory.

### Non-circularity — the WP-3 honesty point (`ctable` vs `ctable_ref`)
The `CompressedScan` plan node carries **two** borrowed pointers:
- `ctable` — the **engine** source. `lower()` builds `tsx::CompressedScan` over `ctable`
  **only**, decoding the compressed bytes independently.
- `ctable_ref` — the **independent** decompressed/source `Table`. The oracle paths read
  this **only**: `render_compressed_scan` registers it as the DuckDB base table (emitting
  the same `SELECT … FROM base` as `render_scan`), and the plan reference treats it as a
  plain scan.

So the differential proves **engine-decode == source** *iff the codec is truly lossless*.
The reference is the **original source held by the test**, never the engine's own
`decode()` output (which would be circular and always agree). Audit checks: `lower()` never
touches `ctable_ref`; the planted decode mutants are caught by the DuckDB diff with the
reference path reading only the source (demonstrated by `compress_mutation_test` /
`catalog_checks_compress`).

### Scalar-by-sequential-dependence vs the vectorized step (rule 3 / D17)
The integer decode runs in three stages: **(1) varint byte-extraction** — inherently
sequential (each value's byte length is data-dependent); **(2) zig-zag decode** —
element-wise, no cross-element dependence ⇒ the SIMD win; **(3) delta-of-delta running
double-prefix-sum** — inherently sequential (value *i* needs value *i−1*). Gorilla (F64)
is end-to-end sequential (variable-length bit blocks XOR'd against the prior value) and
has no vector form. So the **lone genuine vector/scalar twin** is stage (2):
`unzigzag_vec` (Highway `HWY_DYNAMIC_DISPATCH`, in `tsx/compress_kernels.cpp`) and the
independently-written `unzigzag_scalar` (plain C++, separate TU `tsx/compress_scalar.cpp`).
`CompressedScan::set_path(DecodePath)` selects which; `compress_scalar_vector_test` forces
both on identical compressed input and asserts **byte-identical** decoded output. No vector
width / ISA / cache / core-count appears in any signature or body (Highway's
`ScalableTag`/`Lanes` drive the loop).

### Mutation self-test + catalog
`tsx/compress_mutants.{h,cpp}` is a faithful copy of the decode loops over the SAME
`EncodedColumn` format + zig-zag kernel, differing by exactly one step, selected by the
distinctly-named `qe::mutant::CompressMutation`:
- `kDropSecondDerivative` — delta-of-delta omits the second-difference term ⇒ TS/I64 drift.
- `kGorillaLeadingZerosOff` — Gorilla leading-zero count off by one when re-deriving the
  trailing-zero shift ⇒ a reconstructed double wrong by bits (shift stays ≤ 63 — no UB).

`compress_mutation_test` shows the round-trip / compressed-scan diff (reference + DuckDB)
**catches each mutant** while the real decoder passes (non-vacuous `CHECK_FALSE` per
mutant). `catalog_checks_compress.cpp` reproduces the decisive clean-vs-mutant comparison
**decided against DuckDB**, registered as one new catalog `Entry`
(`compress_decode_drift`, hazard `kOperatorOrchestration`). No `Hazard` enumerator added.

---

## Assumptions / ICRs
- **No ICRs.** The plan extension is the pre-authorized additive PLAN-EXTENSION pattern.
- The two timestamp/int codecs assume the I32/I64/TS/F64/BOOL physical layout frozen in
  `core/types.h`; nothing new is asked of any frozen interface.
- Gorilla compresses **smoothly-varying** doubles best (its design target); the bench's
  F64 column uses realistic ±1-cent tick moves, which is representative of tick data.

## Note for the final review (coexistence with WP-13)
The working tree handed to me already contained WP-13's (windowed-aggregation) in-progress
edits to the shared files (`plan/*`, `oracle/*`, `CMakeLists.txt`,
`tests/{catalog_checks.h,mutation_catalog.cpp}`) plus `tsx/window.*`. All my edits to those
shared files are **strictly additive** and appended **after** WP-13's blocks (e.g. the enum
is `{…, AsofJoin, Window, CompressedScan}`); the enum is in-memory only, so the integer
value is irrelevant. To validate WP-14 free of WP-13's mid-flight state I also built it in a
**clean `git worktree` of HEAD with only the WP-14 diff** — green there — and then confirmed
the **integrated** tree (WP-13 + WP-14) builds and all WP-14 tests + the catalog meta-test
pass with DuckDB staged. Nothing in WP-14 depends on WP-13.

---

## Exact commands (host=mac, isa=neon — Mac numbers are preliminary / relative-only, §2)

Configure + build (ASan/UBSan preset) and run the gate + the WP-14 tests:

```bash
# from repo root
cmake --preset asan
cmake --build build-asan -j8

# from-scratch enforcement (no compression library under tsx/ etc.)
scripts/check_forbidden_includes.sh

# 1) pure round-trip — bit-exact incl. NaN/±0/inf/subnormals/nulls (NO DuckDB)
./build-asan/compress_roundtrip_test --seed 20260614

# 2) scalar==vector (force both DecodePaths, assert byte-identical decode)
./build-asan/compress_scalar_vector_test --seed 20260614

# 3) compressed-scan differential vs the independent reference AND DuckDB
#    (DuckDB staged at third_party/duckdb/duckdb.cpp -> QE_WITH_DUCKDB)
./build-asan/compress_differential_test --seed 20260614

# 4) mutation self-test — each planted decode bug is CAUGHT; real decoder passes
./build-asan/compress_mutation_test --seed 20260614

# 5) catalog meta-test (includes the compress_decode_drift entry)
./build-asan/mutation_catalog_meta_test --seed 20260614

# or the whole WP-14 surface at once:
ctest --test-dir build-asan -R 'compress|mutation_catalog_meta' --output-on-failure
```

ASan/UBSan: **green** (re-run above under the `asan` preset). All randomized tests pin
`--seed ${QE_CI_SEED}` (20260614) in CTest and print the seed for replay.

Ratio + decode-throughput (Release build — the §C reported numbers):

```bash
cmake --preset release
cmake --build build --target bench_compress -j8
./build/bench_compress --seed 20260614 --rows 2000000 --reps 11
```

---

## Bench JSON (host=mac, isa=neon, hwy_target=NEON — **preliminary / relative-only**, §2)

`./build/bench_compress --seed 20260614 --rows 2000000 --reps 11`

```json
{"kind":"bench_compress","host":"mac","isa":"neon","hwy_target":"NEON","seed":20260614,
 "preliminary":true,"note":"Mac numbers are preliminary / relative-only (RIGOR.md §2)",
 "rows":2000000,"uncompressed_bytes":48000000,"compressed_bytes":22648209,
 "compression_ratio":2.11937,"decode_rows_per_s_median":3.43926e+07,
 "decode_median_ns":58152125,"decode_reps":11,
 "per_column":[
   {"name":"ts","type":"ts","uncompressed_bytes":16000000,"compressed_bytes":5934724,"ratio":2.696},
   {"name":"px","type":"f64","uncompressed_bytes":16000000,"compressed_bytes":10878452,"ratio":1.4708},
   {"name":"vol","type":"i64","uncompressed_bytes":16000000,"compressed_bytes":5835033,"ratio":2.74206}]}
```

- **Compression ratio** (uncompressed/compressed): **2.12×** overall on the synthetic
  tick stream — TS **2.70×** (delta-of-delta on regular-ish ticks), I64 **2.74×**, F64
  **1.47×** (Gorilla on ±1-cent price moves). Ratios are exact point measurements.
- **Decode throughput:** median **~34.4M rows/s** (3-column tick table; median of 11
  warm full-table drains; `decode_median_ns` is the median, not an average).
- These are Mac/NEON dev numbers — **preliminary / relative-only**; the credible roofline
  and core-scaling come from the x86 box (§2). The exact driver reruns there unchanged
  (nothing hardcodes width/cache/core-count/ISA).
