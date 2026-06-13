# Minimal local build of the VENDORED, pinned Google Highway 1.2.0 core library.
# See VENDORING.md for the pin, license, and the keep/prune rationale. We do NOT
# use upstream's CMakeLists.txt (install/test/packaging machinery we don't need);
# this just compiles the 7 core translation units into a static `hwy` target with
# the include root set so `#include "hwy/highway.h"` resolves.
#
# Header-only consumers (the engine's SIMD kernels, later WPs) need only the
# include directory; the static lib carries the runtime-dispatch tables
# (targets.cc / per_target.cc) the smoke driver and provenance probe link against.

add_library(hwy STATIC
  hwy/abort.cc
  hwy/aligned_allocator.cc
  hwy/nanobenchmark.cc
  hwy/per_target.cc
  hwy/print.cc
  hwy/targets.cc
  hwy/timer.cc)

# Include root is this dir so the upstream "hwy/..." include prefix resolves.
target_include_directories(hwy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Static build: leave HWY_SHARED_DEFINE unset so HWY_DLLEXPORT is empty
# (hwy/highway_export.h). No symbol-visibility export needed.

# Third-party code is NOT held to our -Werror / -Wall -Wextra gate: the engine's
# from-scratch sources are, but vendored Highway must compile clean on every
# toolchain we port to without us patching upstream. Strip the strict flags here.
target_compile_options(hwy PRIVATE -w)

find_package(Threads REQUIRED)
target_link_libraries(hwy PUBLIC Threads::Threads)
