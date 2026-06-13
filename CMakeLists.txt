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

option(ENABLE_SANITIZERS "Build with AddressSanitizer + UBSan" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer (mutually exclusive with ENABLE_SANITIZERS)" OFF)

if(ENABLE_SANITIZERS AND ENABLE_TSAN)
  message(FATAL_ERROR "ENABLE_SANITIZERS (ASan/UBSan) and ENABLE_TSAN cannot be combined")
endif()

add_compile_options(-Wall -Wextra -Werror)

# Never hardcode the ISA flag: detect what the compiler accepts (Apple clang on
# arm64 rejects -march=native; x86 GCC/Clang accept it). Detected at configure
# time, recorded in the provenance string below — config, not a literal.
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

add_compile_definitions(
  QE_GIT_COMMIT="${QE_GIT_COMMIT}"
  QE_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  QE_CXX_FLAGS="${QE_FLAGS_DESC}"
  QE_HWY_TARGET="n/a-WP-H")

# Vendored single-header doctest (third_party/doctest/doctest.h).
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

find_package(Threads REQUIRED)

# The harvested rigor library: histogram + reporting + coordinated-omission
# load harness. Header search root is bench/ (histogram.h, bench_report.h, ...).
add_library(bench_infra STATIC
  bench/histogram.cpp
  bench/bench_report.cpp
  bench/load_harness.cpp)
target_include_directories(bench_infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(bench_infra PUBLIC Threads::Threads)

# Drivers.
add_executable(host_state bench/host_state_main.cpp)
target_link_libraries(host_state PRIVATE bench_infra)

add_executable(bench_skeleton bench/bench_skeleton_main.cpp)
target_link_libraries(bench_skeleton PRIVATE bench_infra)

# Self-tests (the credibility layer).
enable_testing()

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
