# Vendoring DuckDB (oracle backend — TEST/ORACLE ONLY)

The WP-3 differential oracle's **authoritative** backend (decision D16) is DuckDB,
linked **only** into the `qe_oracle` test library and **only** when the
amalgamation is staged here. DuckDB never enters any engine target; the
from-scratch include gate (`scripts/check_forbidden_includes.sh`) covers
`core/ simd/ expr/ ops/ plan/ tsx/` and exempts `oracle/` and `tests/`.

## Why this is a manual pre-stage step

The build sandbox has **no network access** (`curl`/`wget` are denied), so the
worker session could not fetch the amalgamation. The exact artifact to stage is
pinned below; once the three files are present, CMake auto-detects them
(`if(EXISTS third_party/duckdb/duckdb.cpp)`), defines `QE_WITH_DUCKDB`, compiles
the amalgamation, and the oracle tests additionally diff against DuckDB. Until
then the oracle runs its **independent reference backend** and CI stays green.

## Exact artifact to stage (PINNED)

- **Product:** DuckDB C++ amalgamation ("libduckdb-src")
- **Version:** **v1.1.3** (pin; do not float to "latest")
- **Asset:** `libduckdb-src.zip`
- **URL:**
  `https://github.com/duckdb/duckdb/releases/download/v1.1.3/libduckdb-src.zip`

Unzip and place these three files **directly** in this directory:

```
third_party/duckdb/duckdb.hpp
third_party/duckdb/duckdb.h
third_party/duckdb/duckdb.cpp
```

Then record the artifact's checksum here on staging (so the pin is verifiable):

```
# fill in at stage time:
# sha256(libduckdb-src.zip) = <...>
```

> If a different DuckDB version is staged, keep these semantics in mind for the
> comparator: the WP-3 generators are constrained so DuckDB never raises (no
> div/mod, magnitude-bounded integer arithmetic, widening-only casts, finite
> floats), so any 1.x release should diff identically. If you bump the version,
> re-run `oracle_differential_test` and note it in the project change log.

## After staging — reconfigure and run the DuckDB diff

```bash
cmake --preset release            # prints: "DuckDB backend ENABLED"
cmake --build --preset release -j
ctest --preset release -R 'oracle_differential_test|oracle_mutation_test' --output-on-failure
```

The same `qe_oracle` comparator (D11 float epsilon + D12 canonicalization) serves
both backends, so no test code changes when DuckDB is enabled.

## Git hygiene

The amalgamation is large (tens of MB) and is **not** committed by this WP. The
`.gitignore` in this directory keeps the staged sources out of version control;
commit only this `VENDORING.md` and the `.gitignore`.
