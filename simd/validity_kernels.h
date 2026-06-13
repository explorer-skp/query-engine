//  WP-1: validity-bitmap bulk kernels — the first real vector/scalar twins, and
//  the convention (simd/kernel_convention.h) every later kernel author follows.
//
//  Each op below exposes two functions with IDENTICAL signatures:
//    *_vec    — Highway dynamic-dispatch implementation (simd/validity_kernels.cpp)
//    *_scalar — independently-written reference loop (simd/validity_scalar.cpp)
//  They MUST agree on every input; that agreement, plus the boundary tests, is
//  what makes the mutation self-test meaningful.
//
//  All functions take `nbits` (the number of meaningful validity bits) and read
//  words(nbits) 64-bit words. They correctly mask the partial final word — bits
//  past nbits are ignored — which is precisely the step a planted SIMD tail bug
//  drops (see simd/validity_kernels_mutants.h).
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::simd {

// Count of set (valid) bits among the first `nbits` bits.
std::size_t count_set_vec(const std::uint64_t* w, std::size_t nbits);
std::size_t count_set_scalar(const std::uint64_t* w, std::size_t nbits);

// True iff every one of the first `nbits` bits is set (no nulls). nbits == 0 is
// vacuously true.
bool all_valid_vec(const std::uint64_t* w, std::size_t nbits);
bool all_valid_scalar(const std::uint64_t* w, std::size_t nbits);

// True iff at least one of the first `nbits` bits is clear (a null exists).
bool any_null_vec(const std::uint64_t* w, std::size_t nbits);
bool any_null_scalar(const std::uint64_t* w, std::size_t nbits);

// Word-wise AND / OR of two nbits-wide bitmaps into `out` (out = a OP b). Writes
// words(nbits) words. AND is the null-propagation primitive for binary ops
// (result is valid iff both inputs are); OR is its dual. `out` may alias `a` or
// `b`. Padding bits of the final word are written but carry no meaning.
void bit_and_vec(const std::uint64_t* a, const std::uint64_t* b,
                 std::uint64_t* out, std::size_t nbits);
void bit_and_scalar(const std::uint64_t* a, const std::uint64_t* b,
                    std::uint64_t* out, std::size_t nbits);
void bit_or_vec(const std::uint64_t* a, const std::uint64_t* b,
                std::uint64_t* out, std::size_t nbits);
void bit_or_scalar(const std::uint64_t* a, const std::uint64_t* b,
                   std::uint64_t* out, std::size_t nbits);

}  // namespace qe::simd
