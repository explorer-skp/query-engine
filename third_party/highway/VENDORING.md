# Vendored Google Highway — pinned

- **Upstream:** https://github.com/google/highway
- **Pinned release:** `1.2.0` (tag `1.2.0`, `HWY_MAJOR.HWY_MINOR.HWY_PATCH == 1.2.0`,
  see `hwy/base.h`). Released 2024-05-31.
- **License:** Apache-2.0 (`LICENSE`); a few files dual-licensed BSD-3 (`LICENSE-BSD3`).
- **Why vendored, not FetchContent:** the project's self-contained rule (RIGOR.md:
  "This repo is self-contained"). After this one-time vendor, no build step touches the
  network. `third_party/highway/CMakeLists.txt` is a *local minimal* build of the core
  library — we do **not** use upstream's 26 KB `CMakeLists.txt` (install/test/packaging
  machinery we don't need).

## What was kept (core `libhwy` only)

The exact source set upstream compiles into the core `hwy` library
(upstream `CMakeLists.txt` `HWY_SOURCES`):

- All public + private headers under `hwy/` and `hwy/ops/`.
- The 7 core translation units: `abort.cc`, `aligned_allocator.cc`,
  `nanobenchmark.cc`, `per_target.cc`, `print.cc`, `targets.cc`, `timer.cc`.

## What was pruned (not needed by the core library, verified by include-grep)

- `hwy/*_test.cc`, `hwy/tests/`, `hwy/examples/` — test/example targets.
- `hwy/contrib/` — sort/math/dot/image/etc. add-on libraries (`hwy_contrib`).
- Upstream build/packaging: root `CMakeLists.txt`, `BUILD`, `WORKSPACE`,
  `MODULE.bazel`, `*.bazelrc`, `debian/`, `docs/`, `g3doc/`, `.github/`, `cmake/`,
  `*.pc.in`, `run_tests.*`.

The 7 core `.cc` files include only `hwy/` headers (plus a guarded
`sanitizer/common_interface_defs.h`), so the pruned dirs are not reachable from the
core library — checked with `grep -h '#include' <core .cc>`.

## Updating the pin

Re-fetch the new tag's tree, re-run the same keep/prune split, bump the version above,
and re-run `scripts/ci.sh`. Do not hand-edit vendored sources.
