//  WP-4: SCALAR TWIN for the hash-combine kernel (see ops/hash_kernels.h). A
//  plain, Highway-free reference loop — INDEPENDENT code from the Highway path in
//  ops/hash_kernels.cpp (D17: collapsing them would make scalar==vector prove
//  nothing). It computes the splitmix64 finalizer one lane at a time; the
//  scalar==vector differential asserts it agrees with the vector path bit-for-bit
//  on every input, including the SIMD-tail boundary lengths.

#include "ops/hash_kernels.h"

namespace qe::ops {

namespace {
// The from-scratch mixing hash (splitmix64 finalizer), written out plainly here.
inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}
}  // namespace

void hash_combine_scalar(std::uint64_t* acc, const std::uint64_t* words,
                         std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        acc[k] = mix64(acc[k] ^ mix64(words[k]));
    }
}

}  // namespace qe::ops
