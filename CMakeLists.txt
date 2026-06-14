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
  ops/project.cpp)
target_link_libraries(qe_ops PUBLIC qe_expr)

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
  oracle/differential.cpp
  oracle/duckdb_oracle.cpp)
target_link_libraries(qe_oracle PUBLIC qe_ops)

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
