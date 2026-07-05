#!/usr/bin/env bash
# WP-0: one-command gate for the whole repo. Runs, in order, and fails on the
# first red:
#   1. from-scratch enforcement: forbidden-include gate + its bite self-test
#      (proves the gate can actually fail, not just pass);
#   2. configure + build + ctest on all three presets (release, asan/UBSan, tsan)
#      via the WP-H build_and_test.sh — the three WP-H self-tests plus the WP-0
#      header compile-check must all pass under every preset;
#   3. Highway integration smoke: build + run the smoke driver, print the
#      runtime-detected target, and show host_state's provenance JSON now carries
#      the real highway_target (no more "n/a-WP-H").
#
# Every line below is reproducible standalone; this just chains them.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/3] from-scratch gate (forbidden includes) + bite self-test"
scripts/check_forbidden_includes.sh
scripts/check_forbidden_includes.sh --self-test

echo "==> [2/3] build + ctest: release, asan/UBSan, tsan"
scripts/build_and_test.sh

echo "==> [3/3] Highway integration smoke + provenance"
# build_and_test.sh leaves a configured Release tree in build/.
cmake --build --preset release --target hwy_smoke host_state -j >/dev/null
echo "--- hwy_smoke ---"
./build/hwy_smoke
echo "--- host_state provenance JSON (highway_target must be real, not n/a-WP-H) ---"
./build/host_state | tee /dev/stderr | grep -q '"highway_target": *"[^n]' \
  || { echo "host_state provenance missing a real highway_target" >&2; exit 1; }

echo "ci.sh: ALL GATES GREEN"
