# WP-1 Report — Columnar core + SIMD layer

Status: **complete, all gates green** (release + ASan/UBSan + TSan, 8/8 tests each;
`ci.sh` 4/4-style gate green). This WP turns the DRAFT `core/`+`simd/` headers into
frozen contracts and builds the columnar format substrate + the first real
vector/scalar twin kernels.

## What landed

**Frozen headers (banners removed, marked "Frozen at WP-1"):**
- `core/types.h` — unchanged `Type`; **added** free function `byte_width(Type)` (no
  field change to the enum) as the single source of truth for value layout
  (I32=4, I64=8, F64=8, TS=8, BOOL=1 byte — one `uint8` per value, *not* bit-packed).
- `core/column.h` — `Column`/`SelectionVector`/`Batch`/`Schema` public fields
  **unchanged**; documented the view/ownership model and the validity precedence
  (below). No fields added/renamed.
- `simd/kernel_convention.h` — the twin convention every later kernel author
  follows, plus `runtime_vector_bytes()` (the runtime-detected width alignment
  derives from).

**New types I designed (format needs them):**
- `core/buffer.{h,cpp}` — `Buffer`: owning, move-only, aligned, contiguous bytes.
- `core/owned_batch.{h,cpp}` — `OwnedColumn`/`OwnedBatch` (the owning layer) +
  `compact_column` (selection-vector compaction).
- `core/validity.h` — Arrow-style bitmap format primitives (words/mask/get/set/fill).
- `core/selection.h` — selection-vector semantics helpers (`sel_at`, `selected_count`).

**SIMD layer (the only place Highway appears):**
- `simd/aligned_alloc.{h,cpp}` — aligned alloc + `runtime_vector_bytes()` wrapping
  Highway, so `core/` includes **no** Highway headers.
- `simd/validity_kernels.{h,cpp}` (vec) + `simd/validity_scalar.cpp` (twin) —
  `count_set`, `all_valid`, `any_null`, `bit_and`, `bit_or`.
- `simd/gather_kernels.{h,cpp}` (vec) + `simd/gather_scalar.cpp` (twin) —
  `gather32`/`gather64` (Highway `GatherIndex`) + `gather8_scalar`.
- `simd/validity_kernels_mutants.{h,cpp}` — **test-only** planted SIMD-tail mutants.

## Decisions I'm asking you to freeze

**1. Ownership / lifetime model.**
- `Buffer` **owns** its bytes (unique, move-only, frees on destruct). It is the only
  owning primitive.
- `Column`/`Batch`/`SelectionVector` are **non-owning views** (as the frozen contract's
  raw pointers require). A view must not outlive the Buffer(s) it points into.
- Because the frozen `Batch` has no storage field, owning lives in a **separate layer**:
  `OwnedColumn` (owns a data Buffer + optional validity Buffer) and `OwnedBatch` (owns
  the columns + optional selection storage). `view()` produces the frozen views. This
  is how I avoided touching the frozen `Batch` shape — **no ICR needed.**

**2. `all_valid` vs `validity` precedence (D5), now documented in `core/column.h`:**
- `all_valid == true`  ⇒ no nulls; `validity` is **ignored** (canonically `nullptr`).
  Fast path: skip the bitmap.
- `all_valid == false` && `validity != nullptr` ⇒ read per-value nullness from the bitmap.
- `validity == nullptr` **implies** `all_valid == true`. The state
  `(all_valid=false, validity=nullptr)` is **invalid** (asserted in debug; the owning
  layer never produces it). One line: **`all_valid` is authoritative; consult
  `validity` only when it is false.**

**3. Alignment is runtime-derived, not a literal.** `simd::required_alignment_bytes()`
= next power of two ≥ `hwy::VectorBytes()` (Highway's runtime-detected width). `Buffer`
guarantees data aligned to that and **asserts it at allocation**. Highway's allocator
over-aligns to `HWY_ALIGNMENT` (≥ any target vector), so the contract holds on NEON
(16) and AVX-512 (64) with zero source change. No `16`/`64` literal in any alloc path.

**4. Bitmap padding convention.** An all-valid bitmap sets the partial final word's
padding bits to 1 (`fill_all_valid`). Kernels never trust padding: every bulk kernel
masks the final word with `last_word_mask(nbits)`. Tests deliberately set garbage
padding to prove this.

## Rigor / what proves it

- **scalar == vector** for every bitmap kernel and both gather kernels, over thousands
  of seeded random inputs, *plus* a third independent per-bit brute-force reference
  triangulating count/all_valid (`tests/validity_bitmap_test.cpp`).
- **Edge cases** explicitly: all-null, none-null (fast path), single-bit, word
  boundaries 63/64/65/127/128/129/…/257, multiples of 64, partial last word with
  garbage padding, and `nbits==0`.
- **Selection vector**: dense (`sel==nullptr`) vs selected, non-monotonic reorder,
  compaction correctness for data + nulls, BOOL/I32/I64/F64 widths, and **out-of-place**
  compaction (distinct buffers — the §12 aliasing hazard cannot arise; asserted).
- **Mutation self-test** (`tests/validity_mutation_test.cpp`): two planted VECTOR
  tail bugs — `count_set` that forgets to mask the partial word (counts padding) and
  `all_valid` that skips the partial word (misses a null there). The suite SHOWS the
  scalar==vector diff + boundary cases CATCH both, the bug is DORMANT on word-aligned
  lengths (proving it's a genuine *tail* bug), and the REAL kernels pass. A randomized
  sweep confirms the diff catches the mutant 100s of times and never falsely accuses
  the real kernel.
- **Determinism**: seed printed at startup; `--seed N` replays exactly (demonstrated).
- **Sanitizers**: ASan+UBSan and TSan both green (re-run via presets).

## Assumptions
- BOOL is stored 1 byte/value (the validity bitmap is the only bit-packed structure).
- `OwnedBatch` and the views are single-threaded (D15); the `mutable SelectionVector`
  backing in `OwnedBatch::view()` is not thread-shared.
- `gather8` (BOOL, 1-byte lanes) is scalar-only by design — no meaningful SIMD form;
  documented in `simd/gather_kernels.h`, not an omission.

## ICRs
**None.** The owning layer sits beside the frozen view structs, so no frozen field
was touched. `byte_width()` is an additive free function, not a field change.

## Exact commands (all reproducible standalone)

```bash
# Full repo gate (forbidden-include + bite self-test; release+asan+tsan build+ctest;
# Highway smoke + provenance). ~63s on the dev Mac (host=mac, isa=neon).
scripts/ci.sh

# Build + test a single preset (release | asan | tsan):
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    -j && ctest --preset asan
cmake --preset tsan    && cmake --build --preset tsan    -j && ctest --preset tsan

# Just the WP-1 tests (release):
ctest --preset release -R 'buffer_test|validity_bitmap_test|selection_vector_test|validity_mutation_test'

# From-scratch gate clean + PROVE it bites:
scripts/check_forbidden_includes.sh
scripts/check_forbidden_includes.sh --self-test

# Mutation self-test, showing it CATCHES the planted SIMD-tail bug (success lines):
./build/validity_mutation_test -s --seed 42
#   -> count_set: real=65, mutant=128 (counts padding)  => CHECK(buggy != ref) passes
#   -> all_valid: real=false, mutant=true (skips partial word) => caught
#   -> dormant on nbits multiple of 64 (genuine tail bug)

# Replay a randomized test deterministically from its printed seed:
./build/validity_bitmap_test --seed 777      # same seed -> identical run
./build/validity_bitmap_test                 # no seed -> random, prints it for replay
```

Host of the runs above: `host=mac` (Apple Silicon), `isa=neon`, Highway dispatched
target `NEON`. No performance numbers are claimed in WP-1 (relative ratios land with
the bench harness, WP-10); these are correctness/sanitizer gates only.
