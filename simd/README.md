# simd/

Vectorized kernels and their independently-written scalar twins. Portable SIMD via
Google Highway **only** — no raw NEON/AVX intrinsics, no hardcoded width/ISA.

**WP-0 status:** DRAFT convention stub only (`kernel_convention.h`). Frozen at WP-1.
No kernels yet. Highway is build-available (see `third_party/highway/`) but unused.
