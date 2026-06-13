#!/usr/bin/env bash
# ported from raft-rsm/build_and_test.sh — rigor infra, Raft-specifics stripped
#
# One-shot WP gate for the harvested infrastructure: Release build + tests,
# then ASan/UBSan build + tests, then TSan build + tests. TSan is a real gate
# here, not a stub: the coordinated-omission load harness is multithreaded, so
# the self-test exercises it under the race detector.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake --preset release
cmake --build --preset release -j
ctest --preset release

cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan

cmake --preset tsan
cmake --build --preset tsan -j
ctest --preset tsan

echo "build_and_test.sh: all builds and tests passed"
