//  WP-1: SCALAR TWINS for the validity-bitmap kernels (see simd/validity_kernels.h).
//
//  This translation unit is plain, Highway-free C++ and is INTENTIONALLY a
//  separate code path from simd/validity_kernels.cpp (the vector path). It uses
//  std::popcount and straight word loops — not Highway's PopulationCount — so
//  that `scalar == vector` is a comparison of two independent implementations,
//  not a tautology (RIGOR.md rule 3 / decision D17).

#include "simd/validity_kernels.h"

#include <bit>

#include "core/validity.h"

namespace qe::simd {

std::size_t count_set_scalar(const std::uint64_t* w, std::size_t nbits) {
    const std::size_t full = nbits / 64;  // fully-covered words
    std::size_t count = 0;
    for (std::size_t i = 0; i < full; ++i) {
        count += static_cast<std::size_t>(std::popcount(w[i]));
    }
    if (nbits & 63u) {  // partial final word: mask before counting
        const std::uint64_t m = qe::validity::last_word_mask(nbits);
        count += static_cast<std::size_t>(std::popcount(w[full] & m));
    }
    return count;
}

bool all_valid_scalar(const std::uint64_t* w, std::size_t nbits) {
    const std::size_t full = nbits / 64;
    for (std::size_t i = 0; i < full; ++i) {
        if (w[i] != ~0ull) return false;
    }
    if (nbits & 63u) {
        const std::uint64_t m = qe::validity::last_word_mask(nbits);
        if ((w[full] & m) != m) return false;
    }
    return true;
}

bool any_null_scalar(const std::uint64_t* w, std::size_t nbits) {
    return !all_valid_scalar(w, nbits);
}

void bit_and_scalar(const std::uint64_t* a, const std::uint64_t* b,
                    std::uint64_t* out, std::size_t nbits) {
    const std::size_t n = qe::validity::words(nbits);
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] & b[i];
}

void bit_or_scalar(const std::uint64_t* a, const std::uint64_t* b,
                   std::uint64_t* out, std::size_t nbits) {
    const std::size_t n = qe::validity::words(nbits);
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] | b[i];
}

}  // namespace qe::simd
