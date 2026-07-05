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
  inc=$(grep -REin --include=\*.h --include=\*.hpp --include=\*.hh \
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
  lnk=$(grep -REin --include=CMakeLists.txt --include=\*.cmake \
        "(target_link_libraries|link_libraries|find_package|FetchContent)[^#]*(${FORBIDDEN})" \
        "${existing[@]}" 2>/dev/null || true)
  if [[ -n "$lnk" ]]; then
    echo "FORBIDDEN LINK/FIND found in engine module CMake:" >&2
    echo "$lnk" >&2
    hits=1
  fi

  # (3) Forbidden link on any ENGINE target in the ROOT CMakeLists.txt — where
  #     every target in this repo is actually defined (audit H8: the scan above
  #     only reads CMake files UNDER the module dirs, which contain none). A
  #     whole-file grep would false-positive on the legitimate DuckDB links in
  #     test/oracle targets, so this parses add_library/target_link_libraries
  #     blocks and flags forbidden tokens only on targets whose sources live
  #     under the module dirs.
  local root_lnk
  root_lnk=$(python3 - "$FORBIDDEN" "${MODULE_DIRS[@]}" <<'PY'
import re, sys
forb = re.compile(r'(?:' + sys.argv[1] + r')', re.I)
dirs = tuple(d + '/' for d in sys.argv[2:])
try:
    text = open('CMakeLists.txt').read()
except OSError:
    sys.exit(0)
blocks = re.findall(
    r'(?m)^[ \t]*(add_library|target_link_libraries)\s*\(([^)]*)\)', text)
engine = set()
for cmd, body in blocks:
    toks = body.split()
    if cmd == 'add_library' and toks and any(
            t.startswith(dirs) for t in toks[1:]):
        engine.add(toks[0])
hits = []
for cmd, body in blocks:
    toks = body.split()
    if cmd == 'target_link_libraries' and toks and toks[0] in engine:
        for t in toks[1:]:
            if forb.search(t):
                hits.append('CMakeLists.txt: engine target %s links forbidden %r'
                            % (toks[0], t))
print('\n'.join(hits))
PY
)
  if [[ -n "$root_lnk" ]]; then
    echo "FORBIDDEN LINK on an engine target in the root CMakeLists.txt:" >&2
    echo "$root_lnk" >&2
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

  # Plant a forbidden LINK on an engine target in the root CMakeLists.txt (the
  # branch audit H8 found unexercised) — expect the gate to bite, then restore.
  echo "[self-test] 2b/3 planted forbidden root-CMake link — expect the gate to BITE ..."
  cp CMakeLists.txt CMakeLists.txt.selftest_bak
  # shellcheck disable=SC2064
  trap "mv CMakeLists.txt.selftest_bak CMakeLists.txt" EXIT
  printf '\ntarget_link_libraries(qe_core PRIVATE duckdb_amalg)\n' >> CMakeLists.txt
  if scan; then
    echo "[self-test] FAIL: gate did NOT catch the planted root-CMake link" >&2
    exit 1
  fi
  echo "[self-test]   gate bit (nonzero) on the engine-target link."
  mv CMakeLists.txt.selftest_bak CMakeLists.txt
  trap - EXIT

  echo "[self-test] 3/3 probes removed — re-scanning, expect CLEAN ..."
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
