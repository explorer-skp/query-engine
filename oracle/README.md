# oracle/

The differential-testing oracle: the engine's results are compared byte-for-byte
against DuckDB on every query, on both ARM/NEON and x86/AVX-512.

**WP-0 status:** empty skeleton — no headers, no logic yet.

Unlike the engine modules, `oracle/` and `tests/` are the **only** places DuckDB /
SQLite may appear (they arrive in a later WP). The forbidden-include gate
(`scripts/check_forbidden_includes.sh`) therefore exempts this directory.
