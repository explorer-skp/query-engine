//  WP-4: INTERNAL kernel seam for the hash table — NOT a frozen public contract
//  (the analog of expr/kernels.h). This is the genuinely data-parallel, SIMD-
//  worthy part of group-find/probe: the from-scratch MIXING HASH applied in bulk
//  to a column of normalized 64-bit key words and folded into a per-row running
//  accumulator.
//
//  THE FROM-SCRATCH HASH. We use the splitmix64 finalizer
//      mix64(x): x^=x>>30; x*=0xbf58476d1ce4e5b9; x^=x>>27;
//                x*=0x94d049bb133111eb; x^=x>>31;
//  a well-studied avalanche mix that is a BIJECTION on 64 bits. The per-column
//  combine is
//      acc = mix64(acc XOR mix64(word))
//  with acc seeded to the table seed before the first key column. Consequences,
//  documented as the collision contract:
//   * SINGLE 64-bit key: distinct words -> distinct full 64-bit hashes (mix64 is
//     injective; XOR-with-seed and the outer mix64 are too). So a single-column
//     table has NO hash collisions in the full hash space — bucket collisions
//     come ONLY from masking the hash to the power-of-two capacity, which linear
//     probing resolves.
//   * COMPOSITE key: the fold is not injective across the whole tuple, but is
//     well-avalanched; bucket collisions are again resolved by linear probing,
//     and a hash tie still triggers a full key-tuple compare, so it is never a
//     correctness issue — only a probe-length cost.
//
//  TWIN CONVENTION (simd/kernel_convention.h, D17). Two functions, identical
//  signatures, INDEPENDENT code in SEPARATE TUs:
//    hash_combine_vec    — Highway dynamic-dispatch path (ops/hash_kernels.cpp)
//    hash_combine_scalar — hand-written reference loop    (ops/hash_scalar.cpp)
//  They MUST agree on every input; tests/hashtable_scalar_vector_test.cpp proves
//  it directly AND end-to-end (whole insert_or_find/find via each path -> equal
//  group ids). No vector width / ISA appears here; width is Highway's at runtime.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::ops {

// acc[k] = mix64(acc[k] XOR mix64(words[k])) for k in [0, n). Call once per key
// column, after initializing every acc[k] to the table seed, to build the
// composite-key row hash. The scalar twin computes the identical result.
void hash_combine_vec(std::uint64_t* acc, const std::uint64_t* words,
                      std::size_t n);
void hash_combine_scalar(std::uint64_t* acc, const std::uint64_t* words,
                         std::size_t n);

}  // namespace qe::ops
