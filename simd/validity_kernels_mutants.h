//  WP-1: TEST-ONLY planted mutants of the validity-bitmap vector kernels.
//
//  "A checker that cannot fail proves nothing." These are deliberately-broken
//  VECTOR kernels carrying the classic SIMD tail/remainder bug: they process the
//  full-word body correctly but MIS-HANDLE the partial final word. The mutation
//  self-test (tests/validity_mutation_test.cpp) shows the scalar==vector
//  differential and the boundary-word cases CATCH them, then confirms the real
//  kernels pass. They are NOT linked into any engine target — only the mutation
//  test — and exist solely to prove the test suite bites.
#pragma once

#include <cstddef>
#include <cstdint>

namespace qe::simd::mutant {

// BUG: counts the WHOLE final word, never ANDing in last_word_mask(nbits), so
// padding bits (set to 1 in an all-valid bitmap) are counted as real values.
// Identical to the real kernel whenever nbits is a multiple of 64.
std::size_t count_set_vec_tailbug(const std::uint64_t* w, std::size_t nbits);

// BUG: checks only the fully-covered words and SKIPS the partial final word, so
// a null living in the last partial word is missed and the column is wrongly
// reported all-valid. Identical to the real kernel whenever nbits is a multiple
// of 64.
bool all_valid_vec_tailbug(const std::uint64_t* w, std::size_t nbits);

}  // namespace qe::simd::mutant
