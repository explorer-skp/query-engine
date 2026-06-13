# WP-H Report — Rigor-Infrastructure Harvest

One-time harvest of reusable rigor **infrastructure** from the Raft project
(`raft-rsm`) into this engine, stripped of all Raft domain logic. No
query-engine code was written; no Raft consensus/log/replication/RPC/order-book
code or `RIGOR.md` was copied. After the harvest the target builds with **zero**
reference back to the source.

## Source → target map

| Source (`raft-rsm/…`) | Target | Action |
|---|---|---|
| `src/metrics/histogram.{h,cpp}` | `bench/histogram.{h,cpp}` | clean port; ns `rsm::metrics`→`qe::metrics` |
| `bench/bench_report.h` | `bench/bench_report.h` | clean; `MachineState` extended (HAZARD 1) |
| `bench/bench_report.cpp` | `bench/bench_report.cpp` | **HAZARD 1**: macOS sysctl branch added; Linux branch preserved; provenance fields |
| `bench/bench_util.h` | `bench/bench_util.h` | **HAZARD 3**: thread-pinning guarded `__linux__`; PRNG/`nowNs` untouched |
| `bench/load_gen.{h,cpp}` | `bench/load_harness.{h,cpp}` | **HAZARD 2**: coordinated-omission core extracted; work is an injectable `WorkFn` |
| `bench/plot_results.py` | `scripts/plot_results.py` | re-authored generic: percentile/CDF/ratio from JSON; Raft metric names dropped |
| `bench/pick_rate.py` | `scripts/pick_rate.py` | kept (generally useful); glob + field names aligned to skeleton schema |
| `third_party/doctest/doctest.h` | `third_party/doctest/doctest.h` | vendored verbatim (+ provenance line) |
| `CMakeLists.txt` (sanitizer block) | `CMakeLists.txt` + `CMakePresets.json` | minimal; builds only harvested artifacts; Raft subdirs dropped |
| `build_and_test.sh` | `scripts/build_and_test.sh` | Release + ASan/UBSan + **real** TSan gate (harness is threaded) |
| `test/histogram_test.cpp` | `tests/histogram_test.cpp` | clean |
| `test/bench_selftest_test.cpp` | `tests/coord_omission_selftest_test.cpp` | **HAZARD 2** self-test; cluster/leadership cases dropped |
| `test/checker_selftest_test.cpp` | `tests/mutation_selftest_test.cpp` | "planted defect MUST be flagged" kept; Raft linearizability dropped |
| `test/test_main.cpp` | `tests/test_main.cpp` | doctest entry point |
| — | `bench/host_state_main.cpp` | new driver: emits the host-validity JSON |
| — | `bench/bench_skeleton_main.cpp` | new driver: end-to-end exercise emitting per-run JSON |
| — | `.clang-format`, `.clang-tidy` | authored (source had none) to match its style |

**Not harvested** (Raft domain, left behind): `src/raft|rpc|transport|storage|statemachine|client|runtime`,
`faults/linearizability|sim_*|chaos*`, `bench/{bench_cluster,phase7_bench,rsm_bench_main,run_*}`,
`src/metrics/alloc_gate.h`, and every `*raft*/*kv*/*order*/*rpc*` test.

## What was stripped from `load_gen` (HAZARD 2)

The source welded the fixed-rate, intended-send-time correction to
`KvClient`/`PeerMap`/`rpc::NodeId`/`KvStore`/`OrderBook` and an alloc-gate role
tag. Removed all of it. What survives is the rigor mechanism only:

- fixed-rate scheduled-work generator, striped round-robin across T threads;
- each unit's latency measured **from its intended send time** (the CO
  correction), with the **actual-send-time** histogram kept alongside;
- drain cap + send-lateness tracking + scheduled/oks/failures/abandoned counts;
- the "work" is an injectable `WorkFn = std::function<bool(XorShift64&, int)>`
  — no network, no client, no domain object. The seeded per-thread RNG is
  still threaded through so the workload stays reproducible.

The stall self-test (`coord_omission_selftest_test.cpp`) injects a 500 ms stall
into a shared, domain-free `StallableSystem` (the injected work blocks until the
stall deadline) and asserts the **intended** tail reports it while the
**actual** tail hides it. Proven to bite: replacing `record(done - intended)`
with `record(done - send)` collapses `intended p99` from ~470 ms to ~295 ns and
**fails** the suite.

## macOS host-capture fields added (HAZARD 1)

`captureMachineState()` gained an `#ifdef __APPLE__` sysctl branch (Linux branch
preserved intact). Detected at runtime — nothing hardcoded:

- `machdep.cpu.brand_string` → `cpu_model`
- `hw.logicalcpu`, `hw.physicalcpu` → logical / physical counts
- `hw.perflevel0.logicalcpu` / `hw.perflevel1.logicalcpu` → **P / E split**
- `hw.memsize` → `ram_bytes`
- governors `n/a-macOS`, turbo `n/a`, cur-freq `n/a`, thermal `n/a` (honest:
  no cpufreq/intel_pstate/portable SMC on Darwin)
- `getloadavg()` → `loadavg1`; `uname()` → kernel

Plus build provenance carried in the JSON (compile-time, injected by CMake;
ISA **detected** from compiler macros, never hardcoded): `isa`
(`neon`/`avx512`/…), `compiler`, `cxx_flags`, `highway_target` (`n/a-WP-H` until
the SIMD WP lands), `git_commit`. macOS timing is tagged
`timing_validity = preliminary-relative-only`.

## Minimal-CMake decisions for WP-0 to absorb

1. **Arch flag is detected, not hardcoded.** The source's literal
   `-O2 -march=native` breaks older Apple-clang (arm64 rejects `-march=native`).
   `CMakeLists.txt` uses `check_cxx_compiler_flag` to pick `-march=native`
   (x86) → `-mcpu=native` (arm) → empty. (This exact Apple clang 21 happens to
   accept `-march=native`, but the detection keeps older/x86 toolchains and the
   one-source-tree-two-ISAs rule honored.)
2. **Provenance via compile definitions** — `QE_GIT_COMMIT`, `QE_CXX_COMPILER`,
   `QE_CXX_FLAGS`, `QE_HWY_TARGET`. WP-0/SIMD WP should set `QE_HWY_TARGET` from
   the real Highway dispatch once Highway is vendored.
3. **doctest** is the only vendored dependency. Tests are three separate
   executables sharing `tests/test_main.cpp`.
4. **TSan is a real gate**, not a stub — the load harness is multithreaded.
5. **Sanitizer detection fix** (needed for our dev compiler): the source's
   `__SANITIZE_ADDRESS__`/`__SANITIZE_THREAD__` switch is GCC-only; clang only
   exposes `__has_feature`. `coord_omission_selftest_test.cpp` now covers both,
   so the reduced-workload path actually engages under the Mac sanitizer gate.

## Too Raft-entangled to port cleanly

- `bench_cluster.*`, `rsm_bench_main.cpp`, `run_*.sh`, `phase7_*` — these are the
  Raft cluster harness/drivers, not infra; dropped. The skeleton driver replaces
  the *reporting* path they exercised.
- `faults/linearizability.*` and the Raft invariant checkers — domain logic. The
  mutation self-test re-expresses the *mechanism* (a checker fed a planted defect
  must flag it) over the harvested measurement infra instead.
- `src/metrics/alloc_gate.h` (`setThreadAllocRole`) — left behind; not in scope
  and not needed once the workload is injectable.

## Interface Change Requests

None.

## Reproduce (this Mac: host=mac-m5, isa=neon; timing preliminary/relative-only)

Canonical (CMake — **not installed in the harvest session**; commands are the
intended interface):

```
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan -j    && ctest --preset asan
cmake --preset tsan    && cmake --build --preset tsan -j     && ctest --preset tsan
scripts/build_and_test.sh           # all three gates in one shot
scripts/host_state.sh --host mac-m5 --isa neon
```

What was actually run to validate (direct clang++, same flags CMake encodes,
because cmake is absent in this session):

```
# Release infra + drivers + tests, -Wall -Wextra -Werror clean; all green
# ASan/UBSan: 3/3 suites SUCCESS   TSan: 3/3 suites SUCCESS (no races)
# CO self-test: intended p99=469.8ms p99.9=503.3ms | actual p99=0.0ms
# CO self-test against a planted CO defect: FAILURE (intended p99 -> 295ns)
# mutation self-test: 3/3 — each checker accepts clean, flags planted defect
# forbidden-symbol grep (excl. provenance + vendored): EMPTY
```

Samples in `bench/samples/`: `sample_host_state.json`, `sample_bench_clean.json`,
`sample_bench_stall.json` (the stall sample shows intended p99≈289 ms vs actual
p99≈0 ms — the correction, captured in a saved file).
