# WP-4 Report — Hash Table Substrate

**Scope:** the shared open-addressing hash table under WP-5 (group-by) and WP-6
(join). On acceptance `ops/hashtable.h` becomes a **frozen contract**. Substrate
validated by standalone unit + fuzz + scalar==vector + mutation self-tests (the
DuckDB oracle is wired in by WP-5/WP-6, per the brief).

---

## 1. Public API proposed to freeze (`ops/hashtable.h`)

```cpp
inline constexpr std::uint32_t kNoGroup = UINT32_MAX;          // find() miss + empty-slot marker
inline constexpr std::uint64_t kDefaultHashSeed = 0x51E5'7A11'AB1E'5EED;

enum class NullPolicy { kEqual, kNeverMatch };                  // GROUP BY vs JOIN null semantics
enum class HashPath   { kVector, kScalar };                     // default kVector; twin selector

struct HashTableConfig {                                        // named growth policy, no magic
  std::size_t initial_capacity = 1024;                          // rounded up to pow2 (>= 8)
  double      max_load_factor  = 0.7;                           // clamped to (0, 0.95]
  static constexpr std::size_t kMinCapacity   = 8;
  static constexpr double      kMaxLoadCeiling = 0.95;
};

struct KeyColumns { const Column* cols; std::size_t num_cols; const SelectionVector* sel; };

class HashTable {
  HashTable(std::vector<Type> key_types,
            NullPolicy = kEqual, HashTableConfig = {}, std::uint64_t seed = kDefaultHashSeed);

  void insert_or_find(const KeyColumns&, std::size_t n, std::uint32_t* out_groups,
                      HashPath = kVector);                       // build/group path
  void find(const KeyColumns&, std::size_t n, std::uint32_t* out_groups,
            HashPath = kVector) const;                          // join probe path

  std::size_t num_groups() const noexcept;                      // == next id
  std::size_t capacity()   const noexcept;                      // power of two
  NullPolicy  null_policy() const noexcept;
  std::uint64_t seed()     const noexcept;
  const std::vector<Type>& key_types() const noexcept;

  bool          group_is_null  (std::uint32_t g, std::size_t col) const;  // key-store read-back
  std::uint64_t group_key_word (std::uint32_t g, std::size_t col) const;  //  for WP-5/WP-6
};
```

**Design intent.** The table is a *key → group-id index* plus a *key store* for
read-back. It deliberately stores **no payloads** and runs **no aggregates / no
gather** — those are WP-5/WP-6. It hands them exactly two primitives
(`insert_or_find`, `find`) and a way to materialize each group's key tuple
(`group_key_word` / `group_is_null`). This keeps the frozen surface minimal.

**Group ids are stable.** Ids are assigned `0,1,2,…` in first-seen order and
never change — *including across growth/rehash* (growth re-slots groups but
preserves ids). WP-5 can index per-group aggregate-state arrays by id directly.

---

## 2. Semantics chosen (the documented contract)

### Key equality
- **I32 / I64 / TS / BOOL:** exact bitwise equality. I32 is zero-extended into a
  64-bit key word (low 32 bits recover the value); I64/TS bit-cast; BOOL is the
  0/1 byte.
- **F64:** equality on a **canonicalized** bit pattern (so it matches DuckDB
  GROUP BY / join keys, not raw IEEE compare):
  - `-0.0` and `+0.0` → the same key (both canonicalize to `+0.0` bits).
  - every NaN (any sign/payload) → the same key (one canonical quiet NaN). So
    `NaN` groups/joins with `NaN`, whereas raw `==` says `NaN != NaN`.

### NULL keys — **policy is the caller's, not baked in** (`NullPolicy`)
- **`kEqual`** (GROUP BY): a NULL is a value. `NULL == NULL` at the same column
  position, `NULL != non-null`. Null-bearing tuples form findable groups; all
  NULLs in a column group together. Composite `(NULL, 5)` groups with `(NULL,5)`.
- **`kNeverMatch`** (join): a key tuple containing **any** NULL matches nothing,
  not even an identical NULL tuple. `find` of such a key **misses**;
  `insert_or_find` of such a key returns a **fresh, unique, unfindable "dead"
  group id** each time (never deduped, never slotted) — so WP-6 gets exactly
  SQL's "NULLs never join" for both build and probe sides, and the per-row id
  contract still holds.

Per-row null-ness is tracked as a `uint64` mask (one bit per key column ⇒ **up to
63 key columns**, asserted). Equality compares masks first, so a real value that
happens to equal the NULL hash sentinel is still distinguished from a NULL.

---

## 3. Hash function, load factor, growth (none hardcoded as magic)

**From-scratch mixing hash = the splitmix64 finalizer**
```
mix64(x): x^=x>>30; x*=0xbf58476d1ce4e5b9; x^=x>>27; x*=0x94d049bb133111eb; x^=x>>31;
```
applied per key column and folded into a seeded per-row accumulator:
```
acc = seed;  for each key column j:  acc = mix64(acc XOR mix64(word_j));
```
**Collision contract (documented):** `mix64` is a *bijection* on 64 bits, so for a
**single** 64-bit key, distinct keys ⇒ distinct full hashes — bucket collisions
come **only** from masking to the power-of-two capacity, which linear probing
resolves. For **composite** keys the fold is well-avalanched but not injective;
hash ties still trigger a full key-tuple compare, so a collision is never a
correctness issue, only a probe-length cost. The seed is constructor-supplied and
printed by the test harness for deterministic replay.

**Load factor / growth.** `HashTableConfig` carries `initial_capacity` (rounded up
to a power of two, floored at `kMinCapacity = 8`) and `max_load_factor` (clamped
to `(0, kMaxLoadCeiling = 0.95]`, default `0.7`). The table grows (capacity ×2,
re-slot every group from its **stored** hash) once occupied slots exceed
`capacity*max_load_factor`, *after* an insert. Because the factor is `< 1`, a free
slot always exists, so the probe loop can never run forever — directly closing the
§12 "hash-table probe overflow" hazard. No capacity/load/alignment literal appears
in any hot path; all are derived from the named config at construction.

---

## 4. Vectorization & how the scalar twin stays independent

The genuinely data-parallel work in group-find/probe is the **bulk key hashing**;
the slot/probe walk is inherently sequential per row (data-dependent pointer
chasing with variable probe length) and is **shared scalar control flow** in
`hashtable.cpp`, not vectorized. This is honest: open-addressing tables get their
SIMD win from hashing, not the walk.

- **Vector path** (`ops/hash_kernels.cpp`): `hash_combine_vec` — the splitmix64
  fold over `u64` lanes via Highway `HWY_DYNAMIC_DISPATCH`. The two 64-bit
  multiplies use Highway's portable `operator*` for `u64` (native on AVX-512,
  emulated from 32-bit `MulEven` on NEON), so one source compiles to both ISAs.
  No vector width / ISA / alignment is hardcoded — width is `hn::Lanes(d)` with a
  scalar coda for the remainder.
- **Scalar twin** (`ops/hash_scalar.cpp`): `hash_combine_scalar` — an
  independently-written plain loop (D17: separate TU, separate code), not the same
  body selected by a flag.

`HashPath::{kVector,kScalar}` on `insert_or_find`/`find` selects which path
computes the hashes; the probe walk is identical either way. So
`tests/hashtable_scalar_vector_test.cpp` proves agreement **two ways**: directly
(`hash_combine_vec == hash_combine_scalar` over random words at the SIMD-tail
boundary lengths 0,1,…,63,64,65,…,2047,2048) and **end-to-end** (a whole
`insert_or_find`/`find` via each path ⇒ identical group ids).

---

## 5. Mutation self-test (rule 4: a checker that cannot fail proves nothing)

`ops/hashtable_mutants.{h,cpp}` (TEST-ONLY, never linked into an engine target) is
a faithful copy of the real probe/insert/grow loops that **reuses the exact shared
normalization/hash/equality** (`ops/hash_internal.h`), so each mutant differs from
the real table by **exactly one** step in the §12 hotspot:
- **`kSkipRehashOne`** — on growth, one group is not re-slotted (the "skip rehash
  of one element" bug); it becomes unfindable after the table grows.
- **`kFindStopEarly`** — `find` gives up after the first probed slot (stops one
  slot early); collision-displaced keys are reported absent.

`tests/hashtable_mutation_test.cpp` runs both mutants and the real table through
the **same** `std::unordered_map` reference cross-check the fuzz suite uses, and
shows the check **catches every mutant** (`CHECK_FALSE`) while the **real table
passes** (`CHECK`). A dormancy case shows the bugs vanish when no growth / no
probe chain occurs — proving they are genuine growth/probe-walk defects, not
constant errors (a suite testing only tiny inputs would miss them).

---

## 6. Tests & rigor

| Test (ctest name) | What it pins |
|---|---|
| `hashtable_test` | unit: insert-or-find idempotence + dense ids; find miss/hit; find never mutates; composite keys; both NULL policies (incl. composite); F64 -0.0/NaN; key-store read-back; selection vector |
| `hashtable_scalar_vector_test` | scalar==vector for the hash kernel (boundary lengths) **and** end-to-end ids/find across paths |
| `hashtable_fuzz_test` | random insert/find mixes vs `std::unordered_map` ref (all types, both policies, tiny initial capacity to force growth); **collision storm** (5000 keys, cap 8, lf 0.9); **insert past every resize boundary** (all keys still findable after each doubling) |
| `hashtable_mutation_test` | both planted mutants caught; real passes; dormancy proof |

All randomized tests take `--seed N`, print the seed, and CI drives them with the
committed `QE_CI_SEED = 20260614` (the established pattern). Determinism verified:
**60/60 green across 20 random seeds × 3 randomized suites**.

**Sanitizers (re-run, all green):** ASan+UBSan and TSan on all four WP-4 suites.
(The table is single-thread; TSan is trivially clean but run as the gate.)

**From-scratch:** `scripts/check_forbidden_includes.sh` clean over
`core simd expr ops plan tsx`. The only third-party header WP-4 touches is Highway
(allowed), confined to `ops/hash_kernels.cpp` (which opts out of `-Werror` like the
other vector TUs).

---

## 7. Assumptions

1. **Key arity ≤ 63 columns** (the per-row/per-group null mask is a `uint64`
   bit-per-column). Asserted at construction. Far beyond any realistic key width.
2. **Group-id space is `uint32`** (`kNoGroup = UINT32_MAX` reserved as empty/miss
   marker), i.e. up to ~4.29e9 distinct groups; exceeding it is an asserted guard,
   not silent wraparound. Matches the engine's batch/row scale.
3. **`KeyColumns` is caller-assembled.** WP-5/WP-6 build a small array of the key
   Column views (a subset/reorder of a Batch's columns) and pass it with the
   batch's selection vector. All key columns share one physical length + selection.
4. **Read-back is by canonical word.** `group_key_word` returns the *canonical*
   F64 representative (+0.0 for any zero, canonical NaN) — the correct group key
   for output, same as DuckDB's representative.
5. **Substrate, not oracle-wired here.** Per the brief, WP-5/WP-6 wire the table
   into the DuckDB differential; WP-4 ships the standalone rigor above.

---

## 8. ICRs

**None.** No frozen interface (`core/`, `simd/`, `ops/operator.h`, `expr/`) was
modified. WP-4 only *adds* files under `ops/` and new build targets/tests.

One ratification note for the final review (not an ICR): `hash_internal.h`
includes `ops/hashtable.h` for the shared value types (`KeyColumns`, `HashPath`,
constants) — the internal seam depends on the frozen header, not vice-versa, so
the freeze direction is clean.

---

## 9. Exact commands

```bash
# Full repo gate (forbidden-include + bite self-test; build+ctest on release,
# asan/UBSan, tsan; Highway smoke). This is the acceptance gate.
scripts/ci.sh

# From-scratch gate alone (+ prove it bites):
scripts/check_forbidden_includes.sh
scripts/check_forbidden_includes.sh --self-test

# Build + ctest a single preset (release | asan | tsan):
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    -j && ctest --preset asan
cmake --preset tsan    && cmake --build --preset tsan    -j && ctest --preset tsan

# Build the four WP-4 suites (release):
cmake --build --preset release --target \
  hashtable_test hashtable_scalar_vector_test hashtable_fuzz_test hashtable_mutation_test -j

# Run each (CI seed). All print "[doctest] Status: SUCCESS!":
./build/hashtable_test                 --seed 20260614
./build/hashtable_scalar_vector_test   --seed 20260614   # scalar == vector
./build/hashtable_fuzz_test            --seed 20260614   # vs unordered_map ref

# MUTATION self-test — SHOW the reference cross-check catches the planted probes:
./build/hashtable_mutation_test        --seed 20260614
#   -> "kSkipRehashOne mutant detected by the reference cross-check"
#   -> "kFindStopEarly mutant detected by the reference cross-check"
#      (the REAL table passes the identical check in the same run.)

# Replay a fuzz case from its seed (any seed printed by a failing run):
./build/hashtable_fuzz_test            --seed <N>

# Determinism sweep (no --seed => fresh random seed each run, printed for replay):
for i in $(seq 1 20); do ./build/hashtable_fuzz_test; done
```

**Host of record for this run:** `host=mac-m*`, `isa=neon` — development/relative
only (no perf numbers claimed in WP-4; the substrate ships correctness rigor, and
the vec-vs-scalar *ratio* is WP-10's job on the x86 box per §2).
