#!/usr/bin/env bash
# WP-10: ONE script regenerates ALL raw benchmark data (the reproducibility gate,
# RIGOR.md §10). It builds the Release drivers, captures the host-validity record,
# and runs every WP-10 driver into RESULTS_DIR as tagged JSON. Plots regenerate
# SEPARATELY from this saved JSON via scripts/plot_results.py WITHOUT re-running
# the engine — so data capture and reporting are cleanly split.
#
# HOST HONESTY (§2): on macOS every emitted doc carries "preliminary":true; the
# validity gate inside each driver REJECTS a contaminated run (marks valid:false
# and exits non-zero). The SAME script reruns on the x86 box unchanged.
#
# Usage: scripts/run_bench.sh [RESULTS_DIR] [N] [ITERS]
#   RESULTS_DIR  default bench/results
#   N            per-op row count (default 262144)
#   ITERS        timed samples per path (default 200)
#
# Every number this produces ships with its exact command (printed below and in
# the WP report). A deliberately-loaded run is rejected — see the --inject-load
# demo at the end (commented; uncomment to show the gate biting).
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="${1:-bench/results}"
N="${2:-262144}"
ITERS="${3:-200}"
SEED="${SEED:-20260615}"
mkdir -p "$RESULTS_DIR"

echo "==> [1/6] configure + build Release drivers"
cmake --preset release >/dev/null
cmake --build --preset release -j \
  --target host_state bench_ops bench_sort bench_sustained \
           bench_engine_vs_duckdb >/dev/null

echo "==> [2/6] host-validity record"
./build/host_state >"$RESULTS_DIR/host_state.json"

echo "==> [3/6] vec-vs-scalar per operator (expr/hashtable/aggregate/sort)"
./build/bench_ops --op all --n "$N" --iters "$ITERS" --seed "$SEED" \
  --outdir "$RESULTS_DIR" >/dev/null || \
  echo "    (bench_ops exited non-zero: validity gate REJECTED this run)"

echo "==> [4/6] permutation-gather microbench (WP-7 worked example)"
./build/bench_sort --n 1048576 --iters "$ITERS" --seed "$SEED" \
  --out "$RESULTS_DIR/bench_sort_gather.json" >/dev/null

echo "==> [5/6] engine-vs-DuckDB ratio (one Plan, both backends)"
./build/bench_engine_vs_duckdb --n 200000 --iters 40 --seed "$SEED" \
  --outdir "$RESULTS_DIR" >/dev/null

echo "==> [6/6] sustained coordinated-omission (clean + injected-stall)"
./build/bench_sustained --window-s 4 --seed "$SEED" \
  --outdir "$RESULTS_DIR" >/dev/null
./build/bench_sustained --window-s 4 --inject-stall-ms 500 --stall-at-s 1.5 \
  --seed "$SEED" --outdir "$RESULTS_DIR" >/dev/null

echo ""
echo "raw data in $RESULTS_DIR/  ->  regenerate plots+summary (engine NOT re-run):"
echo "    python3 scripts/plot_results.py $RESULTS_DIR"
echo ""
echo "to SHOW the validity gate reject a contaminated run:"
echo "    ./build/bench_ops --op expr --inject-load   # -> valid:false, exit 3"
