//  WP-1: validity-bitmap format primitives (Arrow-style, decision D5).
//
//  Layout: 1 bit per value, packed LSB-first into 64-bit little-endian words.
//  Value i lives at bit (i % 64) of word (i / 64); 1 == valid, 0 == null. A
//  bitmap covering `nbits` values occupies words(nbits) words; bits past nbits in
//  the final partial word are PADDING and carry no meaning.
//
//  This header holds the cheap, inline, scalar-only primitives (single-bit
//  get/set, word counting, the partial-last-word mask). The BULK ops that have a
//  vector + scalar twin — popcount-count, all-valid, AND/OR — live in
//  simd/validity_kernels.h, because those are the ones worth vectorizing and the
//  ones the mutation self-test targets.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::validity {

// Number of 64-bit words needed to hold `nbits` validity bits.
inline constexpr std::size_t words(std::size_t nbits) {
    return (nbits + 63) / 64;
}

// Mask selecting the meaningful (in-range) bits of the FINAL word of an
// nbits-wide bitmap. Returns ~0ull when nbits is a positive multiple of 64 (the
// last word is full) and 0 when nbits == 0. This is the value a correct bulk
// kernel ANDs the last word with before counting/checking — the exact step a
// SIMD tail bug omits.
inline constexpr std::uint64_t last_word_mask(std::size_t nbits) {
    const unsigned rem = static_cast<unsigned>(nbits & 63u);
    if (nbits == 0) return 0ull;
    if (rem == 0) return ~0ull;
    return (std::uint64_t{1} << rem) - 1ull;
}

// Read the validity bit for value i. Precondition: i < the bitmap's nbits.
inline bool get_bit(const std::uint64_t* w, std::size_t i) {
    return (w[i >> 6] >> (i & 63u)) & 1ull;
}

// Set the validity bit for value i to `valid`. Precondition: i < nbits.
inline void set_bit(std::uint64_t* w, std::size_t i, bool valid) {
    const std::uint64_t bit = std::uint64_t{1} << (i & 63u);
    if (valid) {
        w[i >> 6] |= bit;
    } else {
        w[i >> 6] &= ~bit;
    }
}

// Initialize an nbits-wide bitmap to all-valid (all meaningful bits 1). Padding
// bits of the final word are set to 1 as well, which is the canonical
// representation: it keeps an all-valid bitmap byte-comparable to ~0 words.
inline void fill_all_valid(std::uint64_t* w, std::size_t nbits) {
    const std::size_t n = words(nbits);
    for (std::size_t k = 0; k < n; ++k) w[k] = ~0ull;
}

// Initialize an nbits-wide bitmap to all-null (all bits 0).
inline void fill_all_null(std::uint64_t* w, std::size_t nbits) {
    const std::size_t n = words(nbits);
    for (std::size_t k = 0; k < n; ++k) w[k] = 0ull;
}

}  // namespace qe::validity
