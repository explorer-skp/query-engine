#!/usr/bin/env bash
# WP-0: from-scratch enforcement gate.
#
# The engine core must be built from scratch: NO query-engine / dataframe library
# (DuckDB, SQLite, Arrow-as-a-library, Polars, DataFusion, Velox, pandas) may be
# #included or linked anywhere under the engine module dirs. (Arrow is a *format*
# reference, never a dependency.) DuckDB/SQLite are allowed ONLY under tests/ and
# oracle/, where the differential oracle lives — those dirs are exempt here.
#
# Exits nonzero (and prints every offending line as path:line:text) if any
# forbidden include/link is found. On a clean tree it exits 0.
#
# Usage:
#   check_forbidden_includes.sh            # scan; nonzero on any hit
#   check_forbidden_includes.sh --self-test  # prove the gate BITES, then restore
set -uo pipefail
cd "$(dirname "$0")/.."

# Engine module dirs the from-scratch rule covers. tests/ and oracle/ are EXEMPT
# (the oracle deliberately links DuckDB/SQLite in a later WP).
MODULE_DIRS=(core simd expr ops plan tsx exec)

# Forbidden library tokens, matched as a path component of an #include or as a
# CMake link/find target. Word-ish boundaries keep false positives down.
FORBIDDEN='duckdb|sqlite3|sqlite|arrow|polars|datafusion|velox|pandas'

# Source extensions scanned for #include; CMake files scanned for link/find.
SRC_GLOB='*.h *.hpp *.hh *.hxx *.inc *.c *.cc *.cpp *.cxx'

# scan: prints offending lines, returns 0 if clean, 1 if any forbidden hit.
scan() {
  local hits=0
  local existing=()
  for d in "${MODULE_DIRS[@]}"; do
    [[ -d "$d" ]] && existing+=("$d")
  done
  # Nothing built yet => nothing to scan => clean.
  [[ ${#existing[@]} -eq 0 ]] && return 0

  # (1) Forbidden #include in any source/header file.
  local inc
  inc=$(grep -REn --include=\*.h --include=\*.hpp --include=\*.hh \
        --include=\*.hxx --include=\*.inc --include=\*.c --include=\*.cc \
        --include=\*.cpp --include=\*.cxx \
        "^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"][^>\"]*(${FORBIDDEN})" \
        "${existing[@]}" 2>/dev/null || true)
  if [[ -n "$inc" ]]; then
    echo "FORBIDDEN INCLUDE(S) found in engine modules:" >&2
    echo "$inc" >&2
    hits=1
  fi

  # (2) Forbidden link/find in any CMake under the module dirs (future-proofing:
  #     module dirs may gain their own CMakeLists in later WPs).
  local lnk
  lnk=$(grep -REn --include=CMakeLists.txt --include=\*.cmake \
        "(target_link_libraries|link_libraries|find_package|FetchContent)[^#]*(${FORBIDDEN})" \
        "${existing[@]}" 2>/dev/null || true)
  if [[ -n "$lnk" ]]; then
    echo "FORBIDDEN LINK/FIND found in engine module CMake:" >&2
    echo "$lnk" >&2
    hits=1
  fi

  return $hits
}

self_test() {
  echo "[self-test] 1/3 scanning current tree — expect CLEAN ..."
  if ! scan; then
    echo "[self-test] FAIL: tree is not clean before planting a probe" >&2
    exit 1
  fi
  echo "[self-test]   clean."

  # Plant a forbidden include in a throwaway file under a module dir.
  local probe="core/__forbidden_probe_DELETEME.cpp"
  # shellcheck disable=SC2064
  trap "rm -f '$probe'" EXIT
  printf '#include <duckdb.h>\nint planted() { return 0; }\n' > "$probe"
  echo "[self-test] 2/3 planted forbidden include at $probe — expect the gate to BITE ..."

  if scan; then
    echo "[self-test] FAIL: gate did NOT catch the planted forbidden include" >&2
    exit 1
  fi
  echo "[self-test]   gate bit (nonzero) and printed the offending line above."

  rm -f "$probe"
  trap - EXIT
  echo "[self-test] 3/3 probe removed — re-scanning, expect CLEAN ..."
  if ! scan; then
    echo "[self-test] FAIL: tree not clean after removing the probe" >&2
    exit 1
  fi
  echo "[self-test]   clean. GATE PROVEN TO BITE."
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

if scan; then
  echo "check_forbidden_includes.sh: clean — no forbidden includes/links in ${MODULE_DIRS[*]}"
  exit 0
else
  echo "check_forbidden_includes.sh: FORBIDDEN dependency detected (see above)" >&2
  exit 1
fi
