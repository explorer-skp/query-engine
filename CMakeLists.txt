# Minimal top-level build for the WP-H harvested rigor infrastructure only.
# It deliberately wires NOTHING of the engine proper (core/simd/expr/ops/plan/
# tsx arrive in later work packages) — it builds the bench/reporting library,
# the host-state and bench-skeleton drivers, and the three self-test suites.
#
# WP-0 ABSORB NOTES (see WP report):
#   * Arch flag is detected, not hardcoded (-march=native on x86, -mcpu=native
#     on Apple Silicon, empty if neither is accepted) — Apple clang rejects
#     -march=native, so the source's literal flag would break the Mac build.
#   * Build provenance (git commit, compiler, flags, Highway target) is injected
#     as compile definitions and surfaced in the host-state JSON; Highway is not
#     a dependency yet, so QE_HWY_TARGET is "n/a-WP-H" until the SIMD WP lands.
#   * doctest is the only vendored dependency (header-only, third_party/).

cmake_minimum_required(VERSION 3.20)
project(query_engine_harvest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Repo root is an include root so engine headers resolve by their module path
# ("core/types.h", "ops/operator.h", "tools/hwy_target.h", ...). This matches the
# include style the frozen DRAFT headers use among themselves.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(ENABLE_SANITIZERS "Build with AddressSanitizer + UBSan" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer (mutually exclusive with ENABLE_SANITIZERS)" OFF)

if(ENABLE_SANITIZERS AND ENABLE_TSAN)
  message(FATAL_ERROR "ENABLE_SANITIZERS (ASan/UBSan) and ENABLE_TSAN cannot be combined")
endif()

add_compile_options(-Wall -Wextra -Werror)

# Never hardcode the ISA flag: detect what the compiler accepts, then pick
# -march=native (typical on x86 GCC/Clang) -> -mcpu=native (typical on arm) ->
# empty. Detected at configure time, recorded in the provenance string below —
# config, not a literal. (Note: our dev compiler, Apple clang 21 on arm64, DOES
# accept -march=native; older Apple clang did not. The detection keeps both that
# history and the x86 build honest from one source tree.)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" QE_HAS_MARCH_NATIVE)
if(QE_HAS_MARCH_NATIVE)
  set(QE_ARCH_FLAG "-march=native")
else()
  check_cxx_compiler_flag("-mcpu=native" QE_HAS_MCPU_NATIVE)
  if(QE_HAS_MCPU_NATIVE)
    set(QE_ARCH_FLAG "-mcpu=native")
  else()
    set(QE_ARCH_FLAG "")
  endif()
endif()

# Benchmarks run from the Release build.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${QE_ARCH_FLAG} -DNDEBUG")

set(QE_FLAGS_DESC "-O2 ${QE_ARCH_FLAG}")
if(ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
  set(QE_FLAGS_DESC "${QE_FLAGS_DESC} -fsanitize=address,undefined")
endif()

if(ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
  set(QE_FLAGS_DESC "${QE_FLAGS_DESC} -fsanitize=thread")
endif()

# --- Build provenance, injected into the reporting layer (HAZARD 1 inputs) ---
execute_process(
  COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short HEAD
  OUTPUT_VARIABLE QE_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE QE_GIT_RES
  ERROR_QUIET)
if(NOT QE_GIT_RES EQUAL 0 OR QE_GIT_COMMIT STREQUAL "")
  set(QE_GIT_COMMIT "uncommitted")
endif()

# QE_HWY_TARGET is a compile-time FALLBACK only. The authoritative value is
# detected at RUNTIME via Highway dispatch (tools/hwy_target.{h,cpp}) and written
# into the host-state/bench JSON by the emitting drivers — a compile constant
# cannot honestly name the dispatched target (esp. when cross-compiling). The old
# "n/a-WP-H" placeholder is gone now that Highway is vendored and wired.
add_compile_definitions(
  QE_GIT_COMMIT="${QE_GIT_COMMIT}"
  QE_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  QE_CXX_FLAGS="${QE_FLAGS_DESC}"
  QE_HWY_TARGET="runtime-detected-see-hwy_target")

# Vendored single-header doctest (third_party/doctest/doctest.h).
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Vendored, pinned Google Highway 1.2.0 (third_party/highway/, self-contained).
# Builds the static `hwy` target; see third_party/highway/CMakeLists.txt and
# third_party/highway/VENDORING.md. No engine kernels use it yet (WP-1 onward).
add_subdirectory(third_party/highway)

find_package(Threads REQUIRED)

# The harvested rigor library: histogram + reporting + coordinated-omission
# load harness. Header search root is bench/ (histogram.h, bench_report.h, ...).
add_library(bench_infra STATIC
  bench/histogram.cpp
  bench/bench_report.cpp
  bench/load_harness.cpp)
target_include_directories(bench_infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(bench_infra PUBLIC Threads::Threads)

# Runtime Highway-target probe (provenance helper, NOT an engine kernel). It is
# the only thing in WP-0 that links Highway; keeping it out of bench_infra means
# the three WP-H self-tests stay Highway-free. -Wno-error because the TU includes
# third-party Highway template machinery we do not patch (our glue is tiny).
add_library(hwy_target_probe STATIC tools/hwy_target.cpp)
target_link_libraries(hwy_target_probe PUBLIC hwy)
target_compile_options(hwy_target_probe PRIVATE -Wno-error)

# --- WP-1: columnar core + SIMD layer ---------------------------------------
# qe_simd is the ONLY place Highway appears in the engine. The vector kernels and
# the aligned allocator include third-party Highway template headers, so those
# TUs (and only those) opt out of -Werror; the scalar twins keep the full gate.
add_library(qe_simd STATIC
  simd/aligned_alloc.cpp
  simd/validity_kernels.cpp
  simd/validity_scalar.cpp
  simd/gather_kernels.cpp
  simd/gather_scalar.cpp)
target_link_libraries(qe_simd PUBLIC hwy)
set_source_files_properties(
  simd/aligned_alloc.cpp
  simd/validity_kernels.cpp
  simd/gather_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")

# qe_core is the columnar format: Buffer + OwnedColumn/OwnedBatch + compaction.
# It pulls in NO Highway headers directly (it goes through qe_simd), keeping the
# from-scratch grep clean; it links qe_simd PUBLIC so dependents get both.
add_library(qe_core STATIC
  core/buffer.cpp
  core/owned_batch.cpp)
target_link_libraries(qe_core PUBLIC qe_simd)

# Test-only planted mutants of the vector kernels (see the mutation self-test).
# NOT linked into any engine target. Highway TU => -Wno-error.
add_library(qe_validity_mutants STATIC simd/validity_kernels_mutants.cpp)
target_link_libraries(qe_validity_mutants PUBLIC hwy)
target_compile_options(qe_validity_mutants PRIVATE -Wno-error)

# --- WP-2: expression IR + vectorized/scalar kernels + evaluation ------------
# qe_expr is the expression engine: the IR + type inference (expr.cpp), the
# tree-walking evaluator that owns NULL propagation (eval.cpp), and the op×type
# kernel matrix as vector/scalar twins. Only the *_kernels.cpp (vector) TUs pull
# in Highway template headers, so only those opt out of -Werror; the scalar
# twins and the IR/eval keep the full gate. Links qe_core (OwnedColumn + WP-1
# validity kernels), which carries qe_simd/hwy transitively.
add_library(qe_expr STATIC
  expr/expr.cpp
  expr/eval.cpp
  expr/arith_kernels.cpp
  expr/arith_scalar.cpp
  expr/compare_kernels.cpp
  expr/compare_scalar.cpp
  expr/logic_kernels.cpp
  expr/logic_scalar.cpp
  expr/cast_kernels.cpp
  expr/cast_scalar.cpp)
target_link_libraries(qe_expr PUBLIC qe_core)
set_source_files_properties(
  expr/arith_kernels.cpp
  expr/compare_kernels.cpp
  expr/logic_kernels.cpp
  expr/cast_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")

# Test-only planted mutants of the expression kernels / null-propagation (see
# the WP-2 mutation self-test). NOT linked into any engine target. Highway TU.
add_library(qe_expr_mutants STATIC expr/expr_kernels_mutants.cpp)
target_link_libraries(qe_expr_mutants PUBLIC qe_core)
target_compile_options(qe_expr_mutants PRIVATE -Wno-error)

# --- WP-3: operators (scan/filter/project) + in-memory Table source ----------
# qe_ops is the pull-based operator layer (ops/operator.h frozen at WP-3). It
# pulls in NO Highway directly (it goes through qe_expr -> qe_core -> qe_simd) and
# NOTHING from oracle/tests, so the from-scratch grep over ops/ stays clean. Full
# -Wall -Wextra -Werror (no third-party headers here).
add_library(qe_ops STATIC
  ops/table.cpp
  ops/scan.cpp
  ops/filter.cpp
  ops/project.cpp
  ops/aggregate.cpp
  ops/agg_kernels.cpp
  ops/agg_scalar.cpp
  ops/join.cpp
  ops/sort.cpp
  ops/sort_internal.cpp)
target_link_libraries(qe_ops PUBLIC qe_expr qe_hashtable)
# Only the vector kernel TU pulls in Highway template headers, so only it opts
# out of -Werror; the scalar twin, the operator, and the internal helpers keep
# the full gate. qe_ops links qe_hashtable (the WP-4 substrate the GROUP BY path
# builds on); qe_hashtable carries qe_core/qe_simd/hwy transitively.
set_source_files_properties(
  ops/agg_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")

# --- WP-12: tsx Phase-2 operators (tsx/asof.h is WP-12's public surface) -------
# qe_tsx is the time-series operator layer. The as-of join REUSES qe_ops (frozen
# Sort + HashTable + the join_internal gather emitters + Table/expr) and pulls in NO
# Highway directly and NOTHING from oracle/tests, so the from-scratch grep over tsx/
# stays clean. Full -Wall -Wextra -Werror (no third-party headers).
# WP-13 (additive) appends tsx/window.cpp to this existing source list (a sibling WP
# also appends a source here — additive, the final review reconciles).
# WP-14 (additive) also appends the time-series codecs + compressed scan here. The
# from-scratch codecs (Gorilla / delta-of-delta / zig-zag / varint) live in
# tsx/compress.cpp; the lone vectorized decode step (zig-zag) is the Highway TU
# tsx/compress_kernels.cpp with its independently-written scalar twin
# tsx/compress_scalar.cpp (rule 3 / D17). Only the Highway TU opts out of -Werror.
add_library(qe_tsx STATIC
  tsx/asof.cpp
  tsx/window.cpp
  tsx/compress.cpp
  tsx/compress_kernels.cpp
  tsx/compress_scalar.cpp)
target_link_libraries(qe_tsx PUBLIC qe_ops)
set_source_files_properties(
  tsx/compress_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")

# Test-only planted mutant of the as-of operator (boundary `<` / nearest-following /
# ignore-a-key / wrong-LEFT-null). NOT linked into any engine target — only the
# WP-12 mutation self-test (and the consolidated catalog meta-test). Reuses qe_ops'
# shared join_internal emitters + frozen Sort/HashTable so it differs from the real
# operator by exactly one merge/emit step.
add_library(qe_asof_mutants STATIC tsx/asof_mutants.cpp)
target_link_libraries(qe_asof_mutants PUBLIC qe_ops)

# WP-13: test-only planted mutant of the Window operator (tumbling bucket-edge off by
# one / sliding frame P+1). Its own library (no collision with the sibling). NOT
# linked into any engine target — only the WP-13 mutation self-test + the catalog
# meta-test. Reuses qe_ops' shared join_internal gather + agg_internal finalizer +
# frozen Sort/HashTable, so it differs from the real operator by exactly one step.
add_library(qe_window_mutants STATIC tsx/window_mutants.cpp)
target_link_libraries(qe_window_mutants PUBLIC qe_ops)

# WP-14 test-only planted mutant of the compressed scan (drop-second-derivative /
# Gorilla-leading-zeros-off). NOT linked into any engine target — only the WP-14
# mutation self-test + the catalog meta-test. Reuses qe_tsx's CompressedTable format
# and zig-zag kernel so it differs from the real decoder by exactly one decode step.
add_library(qe_compress_mutants STATIC tsx/compress_mutants.cpp)
target_link_libraries(qe_compress_mutants PUBLIC qe_tsx)

# --- WP-8: physical plan nodes + dataframe builder API + lowering + print -----
# qe_plan is the engine-side plan layer (plan/plan.h is WP-8's public surface).
# It links qe_ops (Scan/Filter/Project/Aggregate/HashJoin/Sort + Table + expr) and
# qe_tsx (WP-12: AsofJoin lowering), pulls in NO Highway directly and NOTHING from
# oracle/tests, so the from-scratch grep over plan/ stays clean. Full
# -Wall -Wextra -Werror (no third-party headers).
add_library(qe_plan STATIC plan/plan.cpp)
target_link_libraries(qe_plan PUBLIC qe_ops qe_tsx)

# Test-only planted mutant LOWERING (drop filter / swap join sides / reverse agg
# keys / drop sort). NOT linked into any engine target — only the WP-8 mutation
# self-test. Reuses qe_plan's public Plan surface so it differs from Plan::lower()
# by exactly one mis-lowered node.
add_library(qe_plan_mutants STATIC plan/plan_mutants.cpp)
target_link_libraries(qe_plan_mutants PUBLIC qe_plan)

# --- WP-3: DuckDB differential oracle (TEST/ORACLE TARGETS ONLY) -------------
# qe_oracle is the differential harness: result-set compare (D11/D12), the
# independent reference oracle, seeded generators, SQL rendering, and the DuckDB
# backend. It links qe_ops (the engine under test). DuckDB is NEVER linked into
# any engine target — only here, and only when the amalgamation is staged — so
# the from-scratch gate over core/simd/expr/ops stays green either way.
add_library(qe_oracle STATIC
  oracle/result_set.cpp
  oracle/logical_query.cpp
  oracle/reference_oracle.cpp
  oracle/generators.cpp
  oracle/sql_render.cpp
  oracle/plan_sql.cpp
  oracle/differential.cpp
  oracle/duckdb_oracle.cpp)
# Links qe_plan (WP-8): the oracle renders/lowers plan::Plan through it (the
# single source of truth feeding engine + DuckDB).
target_link_libraries(qe_oracle PUBLIC qe_ops qe_plan)

# DuckDB differential backend, auto-enabled IFF the amalgamation is vendored at
# third_party/duckdb/. The sandbox has no network, so the human pre-stages the
# artifact named in third_party/duckdb/VENDORING.md; CMake then sets
# QE_WITH_DUCKDB (PUBLIC -> visible to oracle TUs and to the test TUs that query
# duckdb_available()) and links the compiled amalgamation. Absent it, the oracle
# runs the independent reference backend alone and CI stays green.
set(QE_DUCKDB_SRC "${CMAKE_SOURCE_DIR}/third_party/duckdb/duckdb.cpp")
if(EXISTS "${QE_DUCKDB_SRC}")
  message(STATUS "WP-3 oracle: DuckDB amalgamation FOUND -> DuckDB backend ENABLED")
  add_library(duckdb_amalg STATIC "${QE_DUCKDB_SRC}")
  target_include_directories(duckdb_amalg PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/duckdb)
  # Vendored third-party TU: silence ALL warnings so the global -Werror can't
  # fail on code we do not own (mirrors the Highway -Wno-error treatment).
  target_compile_options(duckdb_amalg PRIVATE -w)
  target_link_libraries(duckdb_amalg PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
  target_compile_definitions(qe_oracle PUBLIC QE_WITH_DUCKDB)
  target_include_directories(qe_oracle PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/duckdb)
  target_link_libraries(qe_oracle PUBLIC duckdb_amalg)
  # The oracle glue TU includes the vendored duckdb.hpp, whose own warnings would
  # trip the global -Werror (same rationale as the Highway/-w exemptions). Suppress
  # -Werror on just this TU so the third-party header can't fail our build.
  # [reviewer integration fix at WP-3 DuckDB-staging; WP-3 worker to ratify.]
  set_source_files_properties(oracle/duckdb_oracle.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-error")
else()
  message(STATUS "WP-3 oracle: DuckDB amalgamation NOT found -> reference backend "
                 "only (stage third_party/duckdb/; see VENDORING.md)")
endif()

# --- WP-10b: optional, additive parallel-execution layer (decision D15) -------
# qe_parallel wraps the FROZEN operators (no interface change) with a morsel-driven
# multithreaded driver (exec/parallel.h). MorselScan (exec/morsel_scan.*) is a
# 64-aligned row-range Scan; the engine composes the frozen ops + plan. It links
# qe_oracle ONLY for the materialized ResultSet currency + drain_operator (NOT for
# DuckDB — qe_parallel never includes DuckDB, so the from-scratch gate over exec/
# stays green). Threads are configuration; no core count / width / ISA is baked.
add_library(qe_parallel STATIC
  exec/morsel_scan.cpp
  exec/parallel.cpp)
target_link_libraries(qe_parallel PUBLIC qe_oracle)
find_package(Threads REQUIRED)
target_link_libraries(qe_parallel PUBLIC Threads::Threads)

# Test-only planted PARALLELISM mutant: a cross-worker partial-aggregate merge that
# DROPS partials (overwrite instead of accumulate) — a bug that bites ONLY with >1
# worker. NOT linked into any engine target — only the WP-10b mutation self-test +
# the catalog meta-test. Derives qe::exec::ParallelEngine and overrides only the
# three merge hooks, so it differs from the real driver by exactly that merge step.
add_library(qe_parallel_mutants STATIC exec/parallel_mutants.cpp)
target_link_libraries(qe_parallel_mutants PUBLIC qe_parallel)

# --- WP-4: hash table substrate (shared by WP-5 aggregation, WP-6 join) -------
# qe_hashtable is the open-addressing linear-probe table + its from-scratch
# mixing hash. Only the vector kernel TU (ops/hash_kernels.cpp) pulls in Highway
# template headers, so only that TU opts out of -Werror; the scalar twin, the
# normalization, and the table itself keep the full gate. Links qe_core (carries
# qe_simd/hwy transitively); pulls in NO oracle/expr code, so the from-scratch
# grep over ops/ stays clean.
add_library(qe_hashtable STATIC
  ops/hashtable.cpp
  ops/hash_internal.cpp
  ops/hash_kernels.cpp
  ops/hash_scalar.cpp)
target_link_libraries(qe_hashtable PUBLIC qe_core)
set_source_files_properties(
  ops/hash_kernels.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")

# Test-only planted mutant of the hash table (broken probe / skipped rehash). NOT
# linked into any engine target — only the WP-4 mutation self-test. Reuses
# qe_hashtable's shared normalization/hash so the mutant differs from the real
# table by exactly one slot/grow step.
add_library(qe_hashtable_mutants STATIC ops/hashtable_mutants.cpp)
target_link_libraries(qe_hashtable_mutants PUBLIC qe_hashtable)

# --- WP-5: test-only planted mutant of the aggregation operator. NOT linked into
# any engine target — only the WP-5 mutation self-test. Reuses qe_ops' shared
# agg_internal helpers so the mutant differs from the real operator by exactly one
# accumulate/finalize/output step (the §5 "skip a null in SUM" family).
add_library(qe_aggregate_mutants STATIC ops/aggregate_mutants.cpp)
target_link_libraries(qe_aggregate_mutants PUBLIC qe_ops)

# --- WP-6: test-only planted mutant of the hash-join operator. NOT linked into
# any engine target — only the WP-6 mutation self-test. Reuses qe_ops' shared
# join_internal helpers (BuildStore + gather emitters) so the mutant differs from
# the real operator by exactly one build / pair-gen / output step.
add_library(qe_join_mutants STATIC ops/join_mutants.cpp)
target_link_libraries(qe_join_mutants PUBLIC qe_ops)

# --- WP-7: test-only planted mutant of the sort operator. NOT linked into any
# engine target — only the WP-7 mutation self-test. Reuses qe_ops' shared
# sort_internal helpers (materialize + gather_rows) so the mutant differs from the
# real operator by exactly one permutation / emit step (DESC-as-ASC, flipped
# null order, radix sign-bit, reversed tie, dropped tail).
add_library(qe_sort_mutants STATIC ops/sort_mutants.cpp)
target_link_libraries(qe_sort_mutants PUBLIC qe_ops)

# Drivers.
add_executable(host_state bench/host_state_main.cpp)
target_link_libraries(host_state PRIVATE bench_infra hwy_target_probe)

add_executable(bench_skeleton bench/bench_skeleton_main.cpp)
target_link_libraries(bench_skeleton PRIVATE bench_infra hwy_target_probe)

# Highway integration smoke driver: runs ONE trivial vectorized op through
# dynamic dispatch and prints the runtime-detected target. Proves Highway builds
# and dispatches on this host. -Wno-error: same third-party-headers rationale.
add_executable(hwy_smoke tools/hwy_smoke.cpp)
target_link_libraries(hwy_smoke PRIVATE hwy)
target_compile_options(hwy_smoke PRIVATE -Wno-error)

# Self-tests (the credibility layer).
enable_testing()

# WP-0: DRAFT interface headers compile and are self-contained. Instantiates no
# behavior — failing to compile is the whole signal. Full -Wall -Wextra -Werror.
add_executable(headers_compile_test tests/headers_compile_test.cpp)
add_test(NAME headers_compile_test COMMAND headers_compile_test)

add_executable(histogram_test tests/test_main.cpp tests/histogram_test.cpp)
target_link_libraries(histogram_test PRIVATE doctest bench_infra)
add_test(NAME histogram_test COMMAND histogram_test)

add_executable(mutation_selftest_test
  tests/test_main.cpp tests/mutation_selftest_test.cpp)
target_link_libraries(mutation_selftest_test PRIVATE doctest bench_infra)
add_test(NAME mutation_selftest_test COMMAND mutation_selftest_test)

add_executable(coord_omission_selftest_test
  tests/test_main.cpp tests/coord_omission_selftest_test.cpp)
target_link_libraries(coord_omission_selftest_test PRIVATE doctest bench_infra)
add_test(NAME coord_omission_selftest_test COMMAND coord_omission_selftest_test)
# Wall-clock self-test (~5s; several-fold longer under sanitizers).
set_tests_properties(coord_omission_selftest_test PROPERTIES TIMEOUT 120)

# --- WP-1 tests. Each uses tests/wp1_test_main.cpp (prints seed; --seed N). ---
add_executable(buffer_test tests/wp1_test_main.cpp tests/buffer_test.cpp)
target_link_libraries(buffer_test PRIVATE doctest qe_core)
add_test(NAME buffer_test COMMAND buffer_test)

add_executable(validity_bitmap_test
  tests/wp1_test_main.cpp tests/validity_bitmap_test.cpp)
target_link_libraries(validity_bitmap_test PRIVATE doctest qe_core)
add_test(NAME validity_bitmap_test COMMAND validity_bitmap_test)

add_executable(selection_vector_test
  tests/wp1_test_main.cpp tests/selection_vector_test.cpp)
target_link_libraries(selection_vector_test PRIVATE doctest qe_core)
add_test(NAME selection_vector_test COMMAND selection_vector_test)

add_executable(validity_mutation_test
  tests/wp1_test_main.cpp tests/validity_mutation_test.cpp)
target_link_libraries(validity_mutation_test
  PRIVATE doctest qe_core qe_validity_mutants)
add_test(NAME validity_mutation_test COMMAND validity_mutation_test)

# --- WP-2 tests. Reuse tests/wp1_test_main.cpp (prints seed; --seed N). ------
# CI determinism (D13): ctest runs every randomized WP-2 test with a FIXED,
# committed seed, so "green" is reproducible — not luck of a fresh random seed.
# Discovery/fuzz mode is still available by running the binary with NO --seed
# (a new seed is chosen and printed each run; replay any failure with --seed N).
# QE_CI_SEED is the standing CI seed; QE_PINNED_SEED replays the pinned cast bug.
set(QE_CI_SEED 20260614)
set(QE_PINNED_SEED 14745959599513406255)

add_executable(expr_scalar_vector_test
  tests/wp1_test_main.cpp tests/expr_scalar_vector_test.cpp)
target_link_libraries(expr_scalar_vector_test PRIVATE doctest qe_expr)
add_test(NAME expr_scalar_vector_test
  COMMAND expr_scalar_vector_test --seed ${QE_CI_SEED})

add_executable(expr_null_truthtable_test
  tests/wp1_test_main.cpp tests/expr_null_truthtable_test.cpp)
target_link_libraries(expr_null_truthtable_test PRIVATE doctest qe_expr)
add_test(NAME expr_null_truthtable_test
  COMMAND expr_null_truthtable_test --seed ${QE_CI_SEED})

add_executable(expr_property_test
  tests/wp1_test_main.cpp tests/expr_property_test.cpp)
target_link_libraries(expr_property_test PRIVATE doctest qe_expr)
add_test(NAME expr_property_test
  COMMAND expr_property_test --seed ${QE_CI_SEED})

# Permanent REGRESSION for the pinned F64->I64 cast tail/rounding bug: replays the
# exact failing tree from the seed that first exposed it. Fails before the
# RoundAway fix (cast_kernels.cpp), passes after — so the bug cannot silently
# return. (The direct value-level regression lives in expr_scalar_vector_test.)
add_test(NAME expr_property_pinned_castbug
  COMMAND expr_property_test --seed ${QE_PINNED_SEED})

add_executable(expr_mutation_test
  tests/wp1_test_main.cpp tests/expr_mutation_test.cpp)
target_link_libraries(expr_mutation_test PRIVATE doctest qe_expr qe_expr_mutants)
add_test(NAME expr_mutation_test
  COMMAND expr_mutation_test --seed ${QE_CI_SEED})

# --- WP-3 tests. Reuse tests/wp1_test_main.cpp (prints seed; --seed N). -------
# All three link qe_oracle (which carries qe_ops -> qe_expr -> qe_core/simd, and,
# when staged, the DuckDB backend). CI drives them with the committed QE_CI_SEED.
add_executable(ops_test tests/wp1_test_main.cpp tests/ops_test.cpp)
target_link_libraries(ops_test PRIVATE doctest qe_oracle)
add_test(NAME ops_test COMMAND ops_test --seed ${QE_CI_SEED})

# The M1 proof: first end-to-end scan->filter->project diffs green vs the oracle.
add_executable(oracle_differential_test
  tests/wp1_test_main.cpp tests/oracle_differential_test.cpp)
target_link_libraries(oracle_differential_test PRIVATE doctest qe_oracle)
add_test(NAME oracle_differential_test
  COMMAND oracle_differential_test --seed ${QE_CI_SEED})

# Mutation self-test #1: the differential MUST flag a corrupted filter / scan.
add_executable(oracle_mutation_test
  tests/wp1_test_main.cpp tests/oracle_mutation_test.cpp)
target_link_libraries(oracle_mutation_test PRIVATE doctest qe_oracle)
add_test(NAME oracle_mutation_test
  COMMAND oracle_mutation_test --seed ${QE_CI_SEED})

# --- WP-4 tests. Reuse tests/wp1_test_main.cpp (prints seed; --seed N). -------
# CI drives the randomized ones with the committed QE_CI_SEED (D13).
add_executable(hashtable_test tests/wp1_test_main.cpp tests/hashtable_test.cpp)
target_link_libraries(hashtable_test PRIVATE doctest qe_hashtable)
add_test(NAME hashtable_test COMMAND hashtable_test --seed ${QE_CI_SEED})

# scalar==vector for the hash kernel AND end-to-end (find via each path).
add_executable(hashtable_scalar_vector_test
  tests/wp1_test_main.cpp tests/hashtable_scalar_vector_test.cpp)
target_link_libraries(hashtable_scalar_vector_test PRIVATE doctest qe_hashtable)
add_test(NAME hashtable_scalar_vector_test
  COMMAND hashtable_scalar_vector_test --seed ${QE_CI_SEED})

# Fuzz cross-check vs a std::unordered_map reference model (collision storms,
# growth past every resize boundary, random insert/find mixes).
add_executable(hashtable_fuzz_test
  tests/wp1_test_main.cpp tests/hashtable_fuzz_test.cpp)
target_link_libraries(hashtable_fuzz_test PRIVATE doctest qe_hashtable)
add_test(NAME hashtable_fuzz_test COMMAND hashtable_fuzz_test --seed ${QE_CI_SEED})

# Mutation self-test: the reference cross-check MUST flag a broken probe / skipped
# rehash, and pass on the real table.
add_executable(hashtable_mutation_test
  tests/wp1_test_main.cpp tests/hashtable_mutation_test.cpp)
target_link_libraries(hashtable_mutation_test
  PRIVATE doctest qe_hashtable qe_hashtable_mutants)
add_test(NAME hashtable_mutation_test
  COMMAND hashtable_mutation_test --seed ${QE_CI_SEED})

# --- WP-5 tests: hash aggregation / GROUP BY. Reuse tests/wp1_test_main.cpp. ---
# scalar==vector for the masked-reduction kernels (direct + end-to-end global).
add_executable(agg_scalar_vector_test
  tests/wp1_test_main.cpp tests/agg_scalar_vector_test.cpp)
target_link_libraries(agg_scalar_vector_test PRIVATE doctest qe_oracle)
add_test(NAME agg_scalar_vector_test
  COMMAND agg_scalar_vector_test --seed ${QE_CI_SEED})

# The M2 proof: GROUP BY diffs green vs the oracle (reference + DuckDB when staged).
add_executable(aggregate_differential_test
  tests/wp1_test_main.cpp tests/aggregate_differential_test.cpp)
target_link_libraries(aggregate_differential_test PRIVATE doctest qe_oracle)
add_test(NAME aggregate_differential_test
  COMMAND aggregate_differential_test --seed ${QE_CI_SEED})

# Mutation self-test: the differential MUST flag a corrupted aggregation
# (null folded as 0 in SUM / all-null group emits 0 / dropped group), and pass on
# the real operator.
add_executable(agg_mutation_test
  tests/wp1_test_main.cpp tests/agg_mutation_test.cpp)
target_link_libraries(agg_mutation_test
  PRIVATE doctest qe_oracle qe_aggregate_mutants)
add_test(NAME agg_mutation_test
  COMMAND agg_mutation_test --seed ${QE_CI_SEED})

# --- WP-6 tests: hash equi-join. Reuse tests/wp1_test_main.cpp (prints seed). ---
# scalar==vector for the build-row gather kernels (direct) AND end-to-end across
# HashPath x GatherPath.
add_executable(join_scalar_vector_test
  tests/wp1_test_main.cpp tests/join_scalar_vector_test.cpp)
target_link_libraries(join_scalar_vector_test PRIVATE doctest qe_oracle)
add_test(NAME join_scalar_vector_test
  COMMAND join_scalar_vector_test --seed ${QE_CI_SEED})

# The M3 proof: INNER + LEFT join diffs green vs the oracle (reference + DuckDB
# when staged) on seeded random two-table queries + edge cases.
add_executable(join_differential_test
  tests/wp1_test_main.cpp tests/join_differential_test.cpp)
target_link_libraries(join_differential_test PRIVATE doctest qe_oracle)
add_test(NAME join_differential_test
  COMMAND join_differential_test --seed ${QE_CI_SEED})

# Mutation self-test: the differential MUST flag a corrupted join (dropped match /
# wrong LEFT NULL / composite first-key-only / short final batch), and pass on the
# real operator.
add_executable(join_mutation_test
  tests/wp1_test_main.cpp tests/join_mutation_test.cpp)
target_link_libraries(join_mutation_test
  PRIVATE doctest qe_oracle qe_join_mutants)
add_test(NAME join_mutation_test
  COMMAND join_mutation_test --seed ${QE_CI_SEED})

# --- WP-7 tests: sort / ORDER BY. Reuse tests/wp1_test_main.cpp (prints seed). ---
# Two cross-checks that prove something: (1) radix-result == comparison-result on
# the SAME integer inputs (two independent sort algorithms agree), and (2) the
# permutation GATHER is scalar==vector (direct kernel + end-to-end). This is the
# WP that carries the first vec-vs-scalar ratio (see bench_sort).
add_executable(sort_scalar_vector_test
  tests/wp1_test_main.cpp tests/sort_scalar_vector_test.cpp)
target_link_libraries(sort_scalar_vector_test PRIVATE doctest qe_oracle)
add_test(NAME sort_scalar_vector_test
  COMMAND sort_scalar_vector_test --seed ${QE_CI_SEED})

# ORDER BY diffs green vs the oracle (reference + DuckDB when staged) under
# POSITIONAL compare (D12): single/multi key, ASC/DESC mixes, NULLS FIRST/LAST,
# all types, ties, empty input, >2048 rows (multi-batch emit).
add_executable(sort_differential_test
  tests/wp1_test_main.cpp tests/sort_differential_test.cpp)
target_link_libraries(sort_differential_test PRIVATE doctest qe_oracle)
add_test(NAME sort_differential_test
  COMMAND sort_differential_test --seed ${QE_CI_SEED})

# Mutation self-test: under POSITIONAL compare the differential MUST flag a
# corrupted sort (DESC-as-ASC / flipped null order / radix sign-bit / reversed
# tie / dropped tail), and pass on the real operator.
add_executable(sort_mutation_test
  tests/wp1_test_main.cpp tests/sort_mutation_test.cpp)
target_link_libraries(sort_mutation_test
  PRIVATE doctest qe_oracle qe_sort_mutants)
add_test(NAME sort_mutation_test
  COMMAND sort_mutation_test --seed ${QE_CI_SEED})

# WP-7 benchmark: the permutation-gather vec-vs-scalar relative ratio (p50/p99),
# emitted as JSON tagged host+isa. Mac numbers are preliminary / relative-only
# (RIGOR.md §2). Built from the Release config; see the WP report for the command.
add_executable(bench_sort bench/bench_sort_main.cpp)
target_link_libraries(bench_sort PRIVATE qe_ops bench_infra hwy_target_probe)

# --- WP-8 tests: plan + dataframe builder API. Closes M2. ---------------------
# Printable round-trip (no randomness => plain doctest main). Links qe_plan only
# (Table + builder); asserts an exact canonical rendering, stability, and that a
# structural change is visible.
add_executable(plan_print_test tests/test_main.cpp tests/plan_print_test.cpp)
target_link_libraries(plan_print_test PRIVATE doctest qe_plan)
add_test(NAME plan_print_test COMMAND plan_print_test)

# The M2-close proof: the builder reproduces every WP-3..WP-7 shape PLUS a deep
# composite pipeline, each diffing green vs the plan reference (and DuckDB when
# staged) through ONE plan that drives both backends. CI pins QE_CI_SEED (D13).
add_executable(plan_differential_test
  tests/wp1_test_main.cpp tests/plan_differential_test.cpp)
target_link_libraries(plan_differential_test PRIVATE doctest qe_oracle)
add_test(NAME plan_differential_test
  COMMAND plan_differential_test --seed ${QE_CI_SEED})

# Mutation self-test: the plan differential MUST flag a mis-LOWERED plan (dropped
# filter / swapped join sides / reversed agg keys / dropped sort), and pass on the
# correct Plan::lower().
add_executable(plan_mutation_test
  tests/wp1_test_main.cpp tests/plan_mutation_test.cpp)
target_link_libraries(plan_mutation_test
  PRIVATE doctest qe_oracle qe_plan_mutants)
add_test(NAME plan_mutation_test
  COMMAND plan_mutation_test --seed ${QE_CI_SEED})

# --- WP-12 tests: backward as-of join (Phase-2 headline). ---------------------
# The differential: ONE Plan (scan.asof_join) drives both backends — it lowers to
# tsx::AsofJoin and renders to DuckDB ASOF JOIN. Diffs green vs the independent
# brute-force reference (always) and DuckDB (when staged). Links qe_oracle (carries
# qe_plan -> qe_tsx, and DuckDB when staged). CI pins QE_CI_SEED (D13).
add_executable(asof_differential_test
  tests/wp1_test_main.cpp tests/asof_differential_test.cpp)
target_link_libraries(asof_differential_test PRIVATE doctest qe_oracle)
add_test(NAME asof_differential_test
  COMMAND asof_differential_test --seed ${QE_CI_SEED})

# scalar==vector end-to-end (the matched-row gather + the partition hash): force
# each path on identical input and assert byte-identical output (rule 3 / D17).
add_executable(asof_scalar_vector_test
  tests/wp1_test_main.cpp tests/asof_scalar_vector_test.cpp)
target_link_libraries(asof_scalar_vector_test PRIVATE doctest qe_oracle)
add_test(NAME asof_scalar_vector_test
  COMMAND asof_scalar_vector_test --seed ${QE_CI_SEED})

# Mutation self-test: the differential MUST flag a broken as-of (strict boundary /
# nearest-following / ignored partition key / wrong LEFT null), and pass on the real
# operator. Links the planted-mutant library alongside qe_oracle.
add_executable(asof_mutation_test
  tests/wp1_test_main.cpp tests/asof_mutation_test.cpp)
target_link_libraries(asof_mutation_test
  PRIVATE doctest qe_oracle qe_asof_mutants)
add_test(NAME asof_mutation_test
  COMMAND asof_mutation_test --seed ${QE_CI_SEED})

# --- WP-13 tests: windowed / time-bucketed aggregation. -----------------------
# The differential: ONE Plan (scan.window_tumbling / .window_sliding) drives both
# backends — it lowers to tsx::Window and renders to DuckDB (GROUP BY time bucket /
# window function). Diffs green vs the independent reference (always) and DuckDB
# (when staged). Links qe_oracle (carries qe_plan -> qe_tsx, and DuckDB when staged).
add_executable(window_differential_test
  tests/wp1_test_main.cpp tests/window_differential_test.cpp)
target_link_libraries(window_differential_test PRIVATE doctest qe_oracle)
add_test(NAME window_differential_test
  COMMAND window_differential_test --seed ${QE_CI_SEED})

# scalar==vector end-to-end (the sliding child-column gather + the tumbling partition
# hash): force each path on identical input and assert byte-identical output.
add_executable(window_scalar_vector_test
  tests/wp1_test_main.cpp tests/window_scalar_vector_test.cpp)
target_link_libraries(window_scalar_vector_test PRIVATE doctest qe_oracle)
add_test(NAME window_scalar_vector_test
  COMMAND window_scalar_vector_test --seed ${QE_CI_SEED})

# Mutation self-test: the differential MUST flag a broken window (tumbling bucket
# edge off by one / sliding frame P+1) and pass on the real operator. Links the
# planted-mutant library alongside qe_oracle.
add_executable(window_mutation_test
  tests/wp1_test_main.cpp tests/window_mutation_test.cpp)
target_link_libraries(window_mutation_test
  PRIVATE doctest qe_oracle qe_window_mutants)
add_test(NAME window_mutation_test
  COMMAND window_mutation_test --seed ${QE_CI_SEED})

# --- WP-14 tests: time-series compression + compressed scan (Phase-2). --------
# Pure engine-only round-trip: decode(encode(col)) is BIT-EXACT across the full
# float domain (NaN payloads / ±0.0 / ±inf / subnormals) + interleaved nulls, per
# type. No DuckDB. CI pins QE_CI_SEED (D13).
add_executable(compress_roundtrip_test
  tests/wp1_test_main.cpp tests/compress_roundtrip_test.cpp)
target_link_libraries(compress_roundtrip_test PRIVATE doctest qe_tsx)
add_test(NAME compress_roundtrip_test
  COMMAND compress_roundtrip_test --seed ${QE_CI_SEED})

# Compressed-scan differential: ONE Plan (compressed_scan, also + filter/project)
# drives both backends — it lowers to tsx::CompressedScan (decoding the compressed
# bytes) and renders to DuckDB SQL over the INDEPENDENT decompressed reference. Diffs
# green vs the reference (always) and DuckDB (when staged), sweeping batch sizes.
add_executable(compress_differential_test
  tests/wp1_test_main.cpp tests/compress_differential_test.cpp)
target_link_libraries(compress_differential_test PRIVATE doctest qe_oracle)
add_test(NAME compress_differential_test
  COMMAND compress_differential_test --seed ${QE_CI_SEED})
# Heaviest WP-14 test: per-case DuckDB CREATE+INSERT+SELECT under sanitizers. Generous
# wall-clock so a slow staged-DuckDB run never spuriously times out.
set_tests_properties(compress_differential_test PROPERTIES TIMEOUT 600)

# scalar==vector end-to-end for the zig-zag-decode kernel: force each DecodePath on
# identical compressed input and assert byte-identical decoded output (rule 3 / D17).
add_executable(compress_scalar_vector_test
  tests/wp1_test_main.cpp tests/compress_scalar_vector_test.cpp)
target_link_libraries(compress_scalar_vector_test PRIVATE doctest qe_tsx qe_oracle)
add_test(NAME compress_scalar_vector_test
  COMMAND compress_scalar_vector_test --seed ${QE_CI_SEED})

# Mutation self-test: the round-trip / compressed-scan diff MUST flag a broken decode
# (drop-second-derivative / Gorilla-leading-zeros-off), and pass on the real decoder.
add_executable(compress_mutation_test
  tests/wp1_test_main.cpp tests/compress_mutation_test.cpp)
target_link_libraries(compress_mutation_test
  PRIVATE doctest qe_oracle qe_compress_mutants)
add_test(NAME compress_mutation_test
  COMMAND compress_mutation_test --seed ${QE_CI_SEED})

# ============================================================================
# WP-10b: optional parallel-execution path (decision D15). The parallel driver
# (qe_parallel) must produce results IDENTICAL to the single-thread plan path under
# the oracle, at thread counts {1,2,4,8} and several morsel sizes; and TSan must be
# green (the new gate — these targets build under the `tsan` preset like every test,
# no per-test gating). Mac scaling is NOT credible (§2): no scaling number is
# claimed here — the deliverable is CORRECTNESS + TSan-green. CI pins QE_CI_SEED.
# ============================================================================

# Parallel DIFFERENTIAL: every generated plan shape (the WP-8 generators) run through
# the parallel driver == single-thread == reference == DuckDB (when staged), across
# {1,2,4,8} threads and morsel sizes. Links qe_oracle (engine + reference + DuckDB)
# + qe_parallel (the driver under test). DuckDB over many cases under TSan is slow ->
# generous timeout.
add_executable(wp10b_parallel_differential_test
  tests/wp1_test_main.cpp tests/wp10b_parallel_differential_test.cpp)
target_link_libraries(wp10b_parallel_differential_test
  PRIVATE doctest qe_oracle qe_parallel)
add_test(NAME wp10b_parallel_differential_test
  COMMAND wp10b_parallel_differential_test --seed ${QE_CI_SEED})
set_tests_properties(wp10b_parallel_differential_test PROPERTIES TIMEOUT 900)

# Parallel MUTATION self-test: the planted cross-worker merge-drop mutant
# (qe_parallel_mutants) is CLEAN at 1 thread and FLAGGED at 2/4/8 — vs single-thread
# AND reference (and DuckDB when staged). Proves the parallel differential can fail.
add_executable(wp10b_parallel_mutation_test
  tests/wp1_test_main.cpp tests/wp10b_parallel_mutation_test.cpp)
target_link_libraries(wp10b_parallel_mutation_test
  PRIVATE doctest qe_oracle qe_parallel qe_parallel_mutants)
add_test(NAME wp10b_parallel_mutation_test
  COMMAND wp10b_parallel_mutation_test --seed ${QE_CI_SEED})
set_tests_properties(wp10b_parallel_mutation_test PROPERTIES TIMEOUT 300)

# WP-14 ratio + decode-throughput reporting driver (the §5 "reported" numbers).
# Mac/NEON numbers are preliminary / relative-only (§2). Links qe_tsx + bench_infra.
add_executable(bench_compress bench/bench_compress_main.cpp)
target_link_libraries(bench_compress PRIVATE qe_tsx bench_infra hwy_target_probe)

# ============================================================================
# WP-10: benchmark harness (cross-cutting). Measures the finished Phase-1 engine
# credibly — vec-vs-scalar per-operator ratios, engine-vs-DuckDB ratio, a roofline
# framing, a validity gate that REJECTS contaminated runs, and coordinated-
# omission-correct sustained latency. Mac numbers are preliminary / relative-only
# (RIGOR.md §2); the same drivers rerun on x86 UNCHANGED (nothing hardcodes
# width/cache/core-count/ISA). The reusable timing/reporting/gate code joins
# bench_infra so every driver and self-test shares one implementation.
# ============================================================================

# Harness helpers + validity gate join the harvested rigor library. No engine and
# no Highway deps -> bench_infra stays reusable on x86 unchanged.
target_sources(bench_infra PRIVATE
  bench/bench_harness.cpp
  bench/validity_gate.cpp)

# Vec-vs-scalar per-operator driver: drives the expr/hashtable/aggregate/sort path
# seams the operators already expose (no new kernels). Gate-checks before report.
add_executable(bench_ops bench/bench_ops_main.cpp)
target_link_libraries(bench_ops
  PRIVATE qe_plan bench_infra hwy_target_probe)  # qe_plan carries qe_ops

# Engine-vs-DuckDB ratio driver: ONE Plan, both backends. DuckDB is timed only
# when the amalgamation is staged (QE_WITH_DUCKDB, carried by qe_oracle); absent
# it the driver emits engine-only timings and says so.
add_executable(bench_engine_vs_duckdb bench/bench_engine_vs_duckdb_main.cpp)
target_link_libraries(bench_engine_vs_duckdb
  PRIVATE qe_oracle qe_plan bench_infra hwy_target_probe)

# Sustained coordinated-omission driver: a real engine query under open-loop load,
# measured from intended send time; --inject-stall-ms demonstrates the correction.
add_executable(bench_sustained bench/bench_sustained_main.cpp)
target_link_libraries(bench_sustained
  PRIVATE qe_plan bench_infra hwy_target_probe)

# Validity-gate self-test: the gate MUST be shown to bite — static reasons
# (load/thermal/governor) plus the quiescence probe under deliberate
# oversubscription. Links bench_infra only (no engine).
add_executable(validity_gate_test
  tests/test_main.cpp tests/validity_gate_test.cpp)
target_link_libraries(validity_gate_test PRIVATE doctest bench_infra)
add_test(NAME validity_gate_test COMMAND validity_gate_test)
set_tests_properties(validity_gate_test PROPERTIES TIMEOUT 120)

# Coordinated-omission self-test on the REAL engine: an injected stall must move
# the INTENDED-send-time tail while the ACTUAL tail hides it. Links qe_plan (the
# query under load) + bench_infra (the load harness). Wall-clock ~5s.
add_executable(co_engine_selftest_test
  tests/wp1_test_main.cpp tests/co_engine_selftest_test.cpp)
target_link_libraries(co_engine_selftest_test
  PRIVATE doctest qe_plan bench_infra)
add_test(NAME co_engine_selftest_test
  COMMAND co_engine_selftest_test --seed ${QE_CI_SEED})
set_tests_properties(co_engine_selftest_test PROPERTIES TIMEOUT 180)

# ============================================================================
# WP-9: oracle & fuzz framework HARDENING — the consolidated mutation catalog +
# meta-test, the 0-row differential cases, and the compact_column(n==0) regression
# (carry-forward (1) fix lives in core/owned_batch.cpp, signature unchanged). No
# new engine kernel here (scalar-twin rule N/A); DuckDB/SQLite stay confined to
# oracle/ + third_party/ (the forbidden-include gate is unaffected).
# ============================================================================

# REGRESSION for carry-forward (1): compact_column(n==0). Deterministic (no
# randomness) -> plain doctest main; links qe_core only. FAILS (aborts) before the
# core/owned_batch.cpp fix, PASSES after.
add_executable(compact_empty_test
  tests/test_main.cpp tests/compact_empty_test.cpp)
target_link_libraries(compact_empty_test PRIVATE doctest qe_core)
add_test(NAME compact_empty_test COMMAND compact_empty_test)

# 0-row DIFFERENTIAL cases (carry-forward (1) at the differential level): filter-
# selects-nothing, global aggregate over empty input, keyed aggregate over empty,
# empty ORDER BY, empty INNER join — each diffed vs the reference AND DuckDB (when
# staged) through the PLAN path. CI pins QE_CI_SEED (D13).
add_executable(oracle_zero_row_test
  tests/wp1_test_main.cpp tests/oracle_zero_row_test.cpp)
target_link_libraries(oracle_zero_row_test PRIVATE doctest qe_oracle)
add_test(NAME oracle_zero_row_test
  COMMAND oracle_zero_row_test --seed ${QE_CI_SEED})

# THE MUTATION CATALOG META-TEST. The registry (tests/mutation_catalog.*) names
# every must-flag mutant, its §12 hazard, and the test that catches it; the check
# bodies live in conflict-free TUs (tests/catalog_checks_*.cpp — each includes at
# most one of the three distinct qe::mutant::Mutation headers); the three new WP-9
# mutants live in tests/catalog_mutants.cpp. The meta-test asserts >=6 entries,
# every §12 hazard spanned, and — for EVERY entry — the un-mutated engine passes
# and the mutant is flagged. Links qe_oracle (carries the real engine + DuckDB when
# staged) PLUS every mutant library (their qe::mutant::Mutation enums never share a
# TU, only the final link, where they do not interact). CI pins QE_CI_SEED.
add_executable(mutation_catalog_meta_test
  tests/wp1_test_main.cpp
  tests/mutation_catalog_meta_test.cpp
  tests/mutation_catalog.cpp
  tests/catalog_checks_kernel.cpp
  tests/catalog_checks_hash.cpp
  tests/catalog_checks_join_sort.cpp
  tests/catalog_checks_agg_plan.cpp
  tests/catalog_checks_asof.cpp
  tests/catalog_checks_window.cpp
  tests/catalog_checks_compress.cpp
  tests/catalog_checks_parallel.cpp
  tests/catalog_mutants.cpp)
target_link_libraries(mutation_catalog_meta_test PRIVATE
  doctest
  qe_oracle
  qe_validity_mutants
  qe_expr_mutants
  qe_hashtable_mutants
  qe_aggregate_mutants
  qe_join_mutants
  qe_sort_mutants
  qe_plan_mutants
  qe_asof_mutants
  qe_window_mutants
  qe_compress_mutants
  qe_parallel_mutants)
add_test(NAME mutation_catalog_meta_test
  COMMAND mutation_catalog_meta_test --seed ${QE_CI_SEED})
# Runs DuckDB over several queries when staged; generous wall-clock under sanitizers.
set_tests_properties(mutation_catalog_meta_test PROPERTIES TIMEOUT 300)
