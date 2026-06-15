# WP-10 Report — Benchmark harness (cross-cutting)

Measures the finished Phase-1 engine **credibly**: per-operator vec-vs-scalar
ratios, an engine-vs-DuckDB ratio, a roofline framing, a **validity gate** that
rejects contaminated runs, and **coordinated-omission-correct** sustained latency
— all reproducible, all tagged `host`+`isa`, all labeled preliminary on Mac (§2).
No engine kernel was written or changed; the harness **drives existing path
seams**. No frozen interface was touched (**no ICRs**).

## Files delivered (all in my area: `bench/` + `scripts/` + new `tests/`)

| File | Role |
|---|---|
| `bench/validity_gate.{h,cpp}` | host-state + quiescence-probe gate; joins `bench_infra` |
| `bench/bench_harness.{h,cpp}` | shared timing/throughput/roofline/JSON helpers; joins `bench_infra` |
| `bench/bench_ops_main.cpp` | **vec-vs-scalar** driver (expr/hashtable/aggregate/sort seams) |
| `bench/bench_engine_vs_duckdb_main.cpp` | **engine-vs-DuckDB** ratio (one Plan, both backends) |
| `bench/bench_sustained_main.cpp` | **coordinated-omission** latency over a real engine query under load |
| `tests/validity_gate_test.cpp` | proves the gate **bites** (static reasons + probe under load) |
| `tests/co_engine_selftest_test.cpp` | CO correction **catches an injected stall** on the engine |
| `scripts/run_bench.sh` | ONE script regenerates all raw JSON |
| `scripts/plot_results.py` | extended: regenerates summary/plots from saved JSON, **engine not re-run** |
| `CMakeLists.txt` | appended a WP-10 section (my targets only) |

**Scalar-twin: N/A.** WP-10 adds no vectorized kernel — it *drives* the existing
vector paths against their existing independently-written scalar twins. The
gate's quiescence probe is plain scalar arithmetic (no SIMD, width-agnostic).

## JSON schema (doc families, keyed by `kind`)

Every doc carries the WP-7 tag block: `host`, `isa`, `seed`, `preliminary`,
`note`, a `machine_start` MachineState record, a `validity` gate verdict, and a
`results` object with `valid`. Percentiles come from **raw samples** via the
harvested `LatencyHistogram` (HDR-style; p50/p90/p99/p99.9/p99.99 + a dense
quantile grid for CDF regen) — never averaged at capture.

- **`bench_vec_scalar`** (one per operator): `results.vec_ns` / `results.scalar_ns`
  histograms; `speedup_p50|p99|p999_scalar_over_vec`; `throughput` {vec/scalar
  rows_per_s, GB/s, bytes_scanned}; `roofline` {flops, bytes,
  arithmetic_intensity_flops_per_byte, ridge_point_flops_per_byte, classification,
  preliminary}.
- **`bench_engine_vs_duckdb`**: `engine_ns` / `duckdb_ns` histograms;
  `speedup_p50|p99_duckdb_over_engine`; `duckdb_available`; `config.query` =
  `Plan::to_string()`.
- **`bench_sustained`**: `e2e_intended` / `e2e_actual` histograms (intended- vs
  actual-send-time); `co_ratio_p99_intended_over_actual`; `throughput_cps`;
  `injected_stall_ms`; scheduled/oks/abandoned/max_send_lateness.
- **`bench_sort`** (WP-7 worked example, reused): `gather64_vec_ns` /
  `gather64_scalar_ns` + speedup.

## How vec-vs-scalar is driven via the path seams (no new kernels)

Same input, same seed, two paths, ratio:

| Operator | Seam (frozen) | Vector path | Scalar twin |
|---|---|---|---|
| expr | `evaluate(e, batch, Backend)` | `Backend::Vector` | `Backend::Scalar` |
| hashtable | `find(kc, n, out, HashPath)` (const probe) | `HashPath::kVector` | `HashPath::kScalar` |
| aggregate | `Aggregate::set_paths(.., AggKernelPath)` (global reduce) | `kVector` | `kScalar` |
| sort | `Sort::set_gather_path(GatherPath)` (radix forced; whole-row gather) | `kVector` | `kScalar` |

The driver builds one seeded input per operator, warms up (excluded), then records
raw per-call latencies into two histograms and reports the **ratio** so host
variance cancels.

**Honest finding (Mac/NEON):** these columnar-streaming ops are **bandwidth-bound**
(arithmetic intensity 0.025–0.5 flops/byte, all below the assumed ridge), and at
p50 the vector path shows **no win** over the compiler-auto-vectorized scalar twin
(expr 0.80×, hashtable 0.92×, aggregate 0.74×, sort 1.00× — speedup = scalar/vec).
The permutation-gather microbench is the one place vector wins (**1.14×** p50). My
job is to *measure and report this honestly*, not to make the engine faster
(other WPs own the kernels); the wider AVX-512 path may change the story at M5.

## How the validity gate reads host state

Two signals (`bench/validity_gate.cpp`):

1. **Static** — from the runtime-detected `MachineState`: `loadavg1/logical_cpus`
   on every OS; and, where the OS exposes them (x86 Linux), governor must be
   `performance`, intel_pstate turbo known, hottest thermal zone under a ceiling.
   macOS reports these `n/a` and the gate **honestly skips them** (it never mistakes
   an absent knob for a nominal reading — `thermal_checked`/`governor_checked`
   record this).
2. **Quiescence probe** (the portable, macOS-side teeth): a fixed deterministic
   scalar compute kernel timed N times; if `max/min` spread exceeds the ceiling the
   core is not quiescent (contention or thermal throttle stealing cycles unevenly)
   and the run is **rejected**.

A rejected run is still written (`results.valid=false`, reasons listed) but **never
reported as credible**, and the driver exits non-zero so a regen script refuses to
publish it. Nothing hardcodes core count/width/cache/ISA — topology is runtime
sysctl/procfs and the probe is width-agnostic.

## What is / is NOT credible on Mac (§2)

**Delivered on Mac (preliminary / relative-only):** vec-vs-scalar ratios +
percentiles; engine-vs-DuckDB ratio; the validity gate; the coordinated-omission
correction + self-test; full reproducibility. The CO correction and the gate are
**host-independent** (they assert ratios internal to a run), so they are credible
even on Mac.

**NOT claimed on Mac:** no AVX-512 number; no absolute headline; **no credible
memory-bandwidth roofline** — the roofline ridge point is **ASSUMED** (`--ridge`,
default 4.0 flops/byte), labeled `preliminary` in the JSON, so the classification
is *framing*, not a measured roofline. The macOS probe ceiling is looser (2.0 vs
1.35 on x86) because Darwin cannot pin a thread and migrates it across
heterogeneous P/E cores — acknowledged, and exactly why Mac is preliminary. The
engine-vs-DuckDB ratio (272× p50 here) is **ingest-dominated**: `run_plan_duckdb`
reloads the table into a fresh in-memory DuckDB per call (the only oracle seam), so
the number is a relative indicator, not "272× faster than DuckDB". **The roofline,
core-scaling curve, and absolute numbers come from the x86 box at M5** — the same
drivers rerun there with ZERO code changes (the platform-specific bits are runtime
host state, not hardcoded).

## Exact commands (every cited number reproduces in one shot)

Build (Release; DuckDB auto-enabled — amalgamation is staged):
```bash
cmake --preset release
cmake --build --preset release -j
```

Regenerate ALL raw data (one script):
```bash
scripts/run_bench.sh                      # -> bench/results/*.json
```

Regenerate plots + summary from saved JSON (engine NOT re-run):
```bash
python3 scripts/plot_results.py bench/results     # -> bench/results/{summary.txt,plots/}
# (pip install matplotlib for the PNGs; summary.txt needs only stdlib)
```

Reproduce each cited ratio (host=mac-m5, isa=neon, seed printed):
```bash
# vec-vs-scalar p50 (scalar/vec): expr 0.80x hashtable 0.92x aggregate 0.74x sort 1.00x
./build/bench_ops --op all --n 262144 --iters 200 --seed 20260615
# permutation-gather microbench (vector wins): 1.14x p50
./build/bench_sort --n 1048576 --iters 200 --seed 20260615
# engine-vs-DuckDB p50 (ingest-dominated, preliminary): 272.75x
./build/bench_engine_vs_duckdb --n 200000 --iters 30 --seed 20260615
# coordinated omission: clean CO ratio 5.6x ; injected-stall CO ratio 3155x
./build/bench_sustained --window-s 3 --seed 20260615
./build/bench_sustained --window-s 3 --inject-stall-ms 500 --stall-at-s 1.2 --seed 20260615
```

Trigger the validity-gate rejection (deliberately loaded -> REJECTED, valid:false, exit 3):
```bash
./build/bench_ops --op expr --inject-load     # probe spread > ceiling -> rejected
```

Coordinated-omission self-tests (catch an injected stall):
```bash
ctest --preset release -R 'co_engine_selftest_test|coord_omission_selftest_test' --output-on-failure
```

ASan/UBSan gate on the harness (re-run; green):
```bash
cmake --preset asan
cmake --build --preset asan -j --target validity_gate_test co_engine_selftest_test bench_ops
./build-asan/validity_gate_test
./build-asan/co_engine_selftest_test --seed 20260614
./build-asan/bench_ops --op all --n 16384 --iters 5     # UBSan/ASan clean
```
(The engine-vs-DuckDB driver's new code is the same gate/histogram/JSON glue
exercised ASan-clean by `bench_ops`; the DuckDB amalgamation itself is third-party,
built `-w`, and is not part of the from-scratch/sanitizer gate.)

TSan gate on the threaded harness code (the sustained driver lowers a fresh
operator tree per call and runs 6 concurrent read-only queries over one shared
const Table; the gate spawns busy/probe threads) — **green**:
```bash
cmake --preset tsan
cmake --build --preset tsan -j --target validity_gate_test co_engine_selftest_test
./build-tsan/validity_gate_test
./build-tsan/co_engine_selftest_test --seed 20260614   # no data races
```

Forbidden-includes gate (green — bench drivers use only the engine's own headers
and the oracle seam, never DuckDB directly):
```bash
scripts/check_forbidden_includes.sh
```

## Rigor gates — status

- One script regenerates raw data ✓ (`run_bench.sh`); plots regenerate from JSON
  without re-running the engine ✓ (`plot_results.py`).
- A deliberately loaded run is **REJECTED** ✓ (`--inject-load`: probe spread
  7.48 > 2.00, `valid:false`; `validity_gate_test` proves the static + dynamic
  bite).
- CO self-test catches an injected stall ✓ (`co_engine_selftest_test`: intended
  p99 ≈ 465 ms vs actual p99 ≈ 0 ms on the real engine workload).
- Every output tagged host+isa ✓; Mac labeled preliminary ✓; percentiles not
  averages ✓; relative ratios present ✓ (vec-vs-scalar **and** engine-vs-DuckDB).
- ASan/UBSan green on harness code ✓; TSan green on the threaded paths ✓ (re-run
  above). No forbidden includes ✓. Seeds printed & replayable ✓.

## Message for the final review

1. **No ICRs.** WP-10 used only the frozen path seams (expr `Backend`, HashTable
   `HashPath`, Aggregate `AggKernelPath`, Sort `GatherPath`) and the frozen
   plan/oracle entry points. Nothing shared was edited.
2. **Shared file:** I appended a WP-10 section to `CMakeLists.txt` (after the WP-8
   plan tests) and added two sources to the existing `bench_infra` library via
   `target_sources`. WP-9 owns `oracle/` + the mutation catalog; I touched neither.
   Two new ctests registered: `validity_gate_test`, `co_engine_selftest_test`.
3. **Honest headline finding to socialize:** on Mac/NEON the columnar operators are
   **bandwidth-bound and the vector path shows no p50 speedup** over the
   auto-vectorized scalar twin (the gather microbench is the exception at 1.14×).
   This is a *measurement*, not a regression in my WP — it is the kind of result
   the relative harness exists to surface, and it should be re-evaluated on the x86
   AVX-512 box at M5, where the harness reruns unchanged.
4. **Engine-vs-DuckDB caveat:** the 272× p50 ratio is dominated by DuckDB's
   per-call table ingest (the only seam `oracle/duckdb_oracle.h` exposes). For a
   credible execution-only comparison at M5, consider an oracle entry point that
   loads the table once and times only `run`. Flagging as a possible future ICR
   against `oracle/` (WP-9's area) — not needed for WP-10's relative deliverable.
5. **For M5 (x86):** rerun `scripts/run_bench.sh` on the x86 box; the validity gate
   tightens automatically (governor/thermal/turbo become live, probe ceiling 1.35),
   and `--ridge <measured flops/byte>` turns the roofline framing into a credible
   classification. No code changes required.
