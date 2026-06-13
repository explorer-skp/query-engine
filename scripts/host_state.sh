#!/usr/bin/env bash
# Thin wrapper: runs the built host_state driver and prints its JSON validity
# record. Builds the Release driver on demand if it is missing. Any extra args
# (e.g. --host mac-m4 --isa neon) pass straight through.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/host_state
if [[ ! -x "$BIN" ]]; then
  cmake --preset release >/dev/null
  cmake --build --preset release --target host_state -j >/dev/null
fi
exec "$BIN" "$@"
