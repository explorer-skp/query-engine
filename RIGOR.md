# RIGOR.md — the always-on engineering rules

This file is the project's rule layer: every change, in every module, is held to
these standards. Design decisions live in the module READMEs and the WP reports;
this is the non-negotiable part.

## What this project is (the spine)

Everything is a **typed column batch** flowing through composable, **pull-based
vectorized operators** (~2048-value batches in tight, SIMD-friendly loops).
**Correctness** = this engine agreeing with DuckDB byte-for-byte on every query,
on both ARM/NEON and x86/AVX-512, via a self-validated differential oracle.
**Speed** = keeping the batch in cache and proving it with same-machine relative
ratios. This is **not** "a SQL database" — never frame it that way.

## Non-negotiable rules

1. **From-scratch core.** No query-engine or dataframe libraries (DuckDB,
   SQLite, Arrow-as-library, Polars, DataFusion, Velox, pandas) anywhere in
   `core/`, `simd/`, `expr/`, `ops/`, `plan/`, `tsx/`, `exec/`. DuckDB/SQLite
   appear **only** in test/oracle targets. Arrow is a *format reference*, not a
   dependency. Enforced by `scripts/check_forbidden_includes.sh`.
2. **Portable SIMD only.** Use **Google Highway**. No raw NEON or AVX intrinsics
   outside a narrow Highway-guarded kernel. **Never hardcode** vector width,
   cache size, alignment, or core count — detect at runtime or read from config.
   (Developed on Apple Silicon, benchmarked on x86; one source tree must compile
   to both.)
3. **Every vectorized kernel has an independently-written scalar twin** returning
   identical results. Template the kernel × type × op matrix to stay lean, but
   the scalar reference path must be *separate code* from the vector path —
   collapsing them makes `scalar == vector` prove nothing.
4. **The oracle is sacred.** Any new operator/behavior must be wired into the
   DuckDB differential test and must add ≥1 **mutation self-test** (a
   deliberately-broken variant the suite is shown to catch). A checker that
   cannot fail proves nothing. Never shrink the oracle/fuzz/test layer to save
   lines.
5. **Determinism.** All randomness is seeded and the **seed is printed**; every
   fuzz failure must replay from a single command (`--seed N`).
6. **Sanitizers are a gate.** ASan + UBSan must be green before any work package
   is "done" (TSan as well, now that multithreading exists). Re-run them; don't
   assume.
7. **No number without a command.** Every performance or correctness claim ships
   with the exact command that reproduces it. Percentiles (p50/p99/p99.9) and
   **relative ratios** (vectorized-vs-scalar, engine-vs-DuckDB), never bare
   averages.
8. **Host honesty.** Tag every measurement `host=` and `isa=`. **Mac numbers are
   preliminary / relative-only** — no AVX-512 claim, no memory-bandwidth
   roofline, no core-scaling curve from macOS (no governor, no pinning, P/E
   heterogeneity). Headline numbers and the roofline come from the x86 box.

## §2 — the Mac-vs-x86 honesty rules (referenced as "§2" throughout)

Developed on Apple Silicon (ARM64/NEON); headline benchmarks belong to an
x86-64 box (AVX-512). Mac results may claim: logic/operator correctness,
cross-ISA oracle coverage, and *preliminary, directional* relative ratios. Mac
results may NOT claim: AVX-512 numbers, a memory-bandwidth roofline, clean
core-scaling curves, or stable absolute latency. Nothing in the codebase may
hardcode ISA, vector width, cache size, or core count — the x86 rerun must need
zero code changes.

## Interface discipline

The headers that cross module boundaries are **frozen contracts**. Do not edit a
frozen interface to make code fit; a change needs an explicit, recorded
interface-change decision, applied additively wherever possible (`git diff` on
the frozen header must show additions only). Width/ISA/cache/core-count must
never appear in a public signature.

## Float & comparison contract

- Integer/exact ops vs the oracle: **exact equality**. Float aggregates:
  **relative + absolute epsilon** (summation order differs; bit-exactness is
  impossible). The epsilons are documented in `oracle/result_set.h`.
- F64 MIN/MAX/ORDER BY use the **NaN-greatest total order** (matches DuckDB):
  NaN sorts above every value, MIN is NaN only for an all-NaN group, MAX is NaN
  when any input is NaN.
- Unordered results: canonicalize by sorting both result sets on all output
  columns before diff, unless the query has an explicit ORDER BY (then compare
  positionally).

## Definition of done (every work package)

Conforms to the frozen interface · from-scratch (forbidden-include grep clean) ·
scalar twin present · oracle diff + new mutation self-test caught · ASan/UBSan
(+TSan) green, re-run · seeds printed & replayable · measurements tagged
host+isa with reproduce commands · reproducibility scripts still work · short WP
report written.

## Non-goals (deliberate scope cuts)

No SQL parser (dataframe/plan API by design) · no persistence/WAL/storage
engine · no transactions/MVCC · no distributed execution · no cost-based
optimizer. These keep the project about execution *technique*.
