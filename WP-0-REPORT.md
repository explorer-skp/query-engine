# WP-0 Report — Scaffold & Gates

Lays the empty skeleton the engine will be built into, plus the from-scratch
enforcement gate and Google Highway build-availability. **No engine behavior**:
no operator, kernel, expression, or plan logic. Built strictly ON the audited
WP-H baseline (`74f8765`) — no WP-H artifact was rewritten or removed.

## What landed

### 1. Module tree (empty skeleton)
`core/ simd/ expr/ ops/ plan/ tsx/ oracle/` created, each with a tracked
`README.md` placeholder stating its role and "WP-0: empty skeleton". `bench/
scripts/ tests/ third_party/` left as WP-H made them. New `tools/` dir holds the
Highway glue (not an engine module).

### 2. DRAFT interface header stubs (verbatim, NON-BINDING)
Transcribed **verbatim** from the WP-0 brief, each carrying the DRAFT banner:
- `core/types.h` — `enum class Type`
- `core/column.h` — `Column`, `SelectionVector`, `Batch`, `Schema`
- `ops/operator.h` — `class Operator`
- `simd/kernel_convention.h` — the scalar-twin contract, no implementation

All in `namespace qe` (`qe::simd` for the kernel convention) to match WP-H. No
fields added, nothing renamed, no logic. `tests/headers_compile_test.cpp` includes
all four, instantiates nothing (only `sizeof > 0` type-completeness static_asserts),
and is wired into ctest as `headers_compile_test`. It compiles under full
`-Wall -Wextra -Werror`.

### 3. Google Highway (build-available, unused)
**Vendored**, pinned **1.2.0**, into `third_party/highway/` (self-contained rule).
Kept exactly the core `libhwy` source set (all `hwy/` + `hwy/ops/` headers + the 7
core `.cc`: abort, aligned_allocator, nanobenchmark, per_target, print, targets,
timer). Pruned tests/examples/contrib/bazel/debian/docs and upstream's 26 KB
CMakeLists — verified by include-grep that the 7 core `.cc` reach only `hwy/`
headers. A local minimal `third_party/highway/CMakeLists.txt` builds the static
`hwy` target. Provenance + keep/prune rationale in `third_party/highway/VENDORING.md`.

- **Smoke driver** `tools/hwy_smoke.cpp`: runs ONE trivial vectorized op (float
  sum) through `HWY_DYNAMIC_DISPATCH` and prints the runtime-detected target.
- **Real `QE_HWY_TARGET`**: the WP-H `"n/a-WP-H"` placeholder is gone. The
  authoritative target is now **detected at runtime** via `tools/hwy_target.{h,cpp}`
  (`qe::tools::dispatchedHighwayTarget()`, a tiny static lib `hwy_target_probe`
  linking `hwy`) and written into the host-state/bench JSON by the emitting drivers.
  On this Mac the JSON `highway_target` field now reads **`NEON`**.

### 4. From-scratch gate (automated, proven to bite)
`scripts/check_forbidden_includes.sh` greps `core/ simd/ expr/ ops/ plan/ tsx/`
for forbidden `#include`s (and forbidden CMake link/find) of duckdb / sqlite /
arrow / polars / datafusion / velox / pandas; `tests/` and `oracle/` are exempt.
Exits nonzero on any hit. `--self-test` **demonstrates the gate failing**: it
plants `#include <duckdb.h>` in a throwaway file under `core/`, shows the gate
catching it (nonzero + the offending `path:line:text`), removes it, and confirms
clean again. On today's tree the plain scan is clean (exit 0).

### 5. One-command gate `scripts/ci.sh`
Runs, failing on first red: (1) forbidden-include gate + its bite self-test;
(2) WP-H `build_and_test.sh` (configure+build+ctest on release, asan/UBSan, tsan);
(3) Highway smoke + the real `highway_target`. `build_and_test.sh` itself is
unchanged.

### 6. WP-H follow-ups absorbed
- **Arch flag stays detected**, not hardcoded — `check_cxx_compiler_flag`
  (`-march=native` → `-mcpu=native` → empty) preserved untouched.
- **`QE_HWY_TARGET` wired to the real Highway target** (item 3 above).
- **Stale comment corrected**: the CMake note claiming Apple clang rejects
  `-march=native` now states the truth on clang 21 (it accepts it) while keeping
  the older-toolchain/x86 rationale for the detection.

## Design notes / assumptions

- **Runtime, not compile-time, target.** A compile constant cannot honestly name
  the dispatched SIMD target (and would be wrong under cross-compilation). So
  `QE_HWY_TARGET` is now only a fallback string (`"runtime-detected-see-hwy_target"`)
  and the JSON-emitting drivers (`host_state`, `bench_skeleton`) overwrite
  `MachineState::hwyTarget` with the runtime value before serializing.
- **`bench_infra` stays Highway-free on purpose.** Only `tools/` links Highway;
  the three WP-H self-tests do not, so the existing gate carries zero new risk
  from the SIMD dependency. The two drivers gained a link to `hwy_target_probe`.
- **Repo root is an include root** (`include_directories(${CMAKE_SOURCE_DIR})`) so
  the DRAFT headers' module-path includes (`"core/types.h"`, `"tools/hwy_target.h"`)
  resolve. This matches the include style baked into the verbatim signatures.
- **Vendored Highway is not held to `-Werror`** (`-w` on the `hwy` target, and
  `-Wno-error` on the two Highway-glue TUs in `tools/`): third-party template
  machinery must compile on every toolchain without us patching upstream. Our own
  engine sources remain under full `-Wall -Wextra -Werror`.

## Confirmation: WP-H self-tests still pass (hard gate)
`headers_compile_test` + `histogram_test` + `mutation_selftest_test` +
`coord_omission_selftest_test` = **4/4 pass on all three presets** (release,
asan/UBSan, tsan). The three WP-H tests are unchanged and green; the CO self-test
still reports intended-p99 ≈ 9.3–9.8 s wall under the stall path.

host=mac-m5, isa=neon. Mac timing is preliminary/relative-only per host-honesty.

## Interface Change Requests
**None.** All four DRAFT signatures were transcribed verbatim; no deviation.

## Reproduce — exact commands

```bash
# (a) Full one-command gate: forbidden-include gate + bite self-test, all three
#     presets build+ctest, Highway smoke + real target.
scripts/ci.sh

# (b) Each preset individually (4/4 tests each: 3 WP-H + headers_compile_test).
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan -j    && ctest --preset asan
cmake --preset tsan    && cmake --build --preset tsan -j    && ctest --preset tsan

# (c) Forbidden-include gate: clean pass, then SHOW IT BITE on a planted include.
scripts/check_forbidden_includes.sh            # -> clean, exit 0
scripts/check_forbidden_includes.sh --self-test  # -> plants <duckdb.h>, catches it, restores

# (d) Highway smoke + its detected target.
cmake --build --preset release --target hwy_smoke -j
./build/hwy_smoke              # highway_dispatched_target=NEON, version 1.2.0, sum check
./build/hwy_smoke --target-only  # -> NEON

# (e) host_state now carries the REAL highway_target (not n/a-WP-H).
./build/host_state --host mac-m5 --isa neon | python3 -c \
  "import sys,json;print('highway_target =',json.load(sys.stdin)['machine']['highway_target'])"
# -> highway_target = NEON
```

Observed on this Mac (host=mac-m5, isa=neon, Apple clang 21, cmake 4.3.3):
`highway_dispatched_target=NEON`, `highway_static_target=NEON_WITHOUT_AES`
(Release) / `NEON` (Debug presets), `highway_version=1.2.0`, smoke sum
10000.0/10000.0. All gates green. **QE_HWY_TARGET on this Mac = `NEON`.**
