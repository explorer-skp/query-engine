//  WP-1: MUTATION SELF-TEST. "A checker that cannot fail proves nothing."
//
//  simd/validity_kernels_mutants.{h,cpp} carry deliberately-broken VECTOR kernels
//  with the classic SIMD tail bug: they handle the full-word body but MIS-HANDLE
//  the partial final word. Here we DEMONSTRATE that the same differential the
//  real suite uses (scalar == vector) plus the boundary-word cases CATCH the
//  planted bug — and that the REAL kernels pass those exact checks. We also show
//  the bug is dormant on word-aligned lengths, proving it is a genuine *tail* bug
//  rather than a constant error.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/validity.h"
#include "simd/validity_kernels.h"
#include "simd/validity_kernels_mutants.h"
#include "tests/wp1_seed.h"

using namespace qe;
using qe::validity::words;

TEST_CASE("MUTATION: count_set tail bug is CAUGHT by the scalar==vector diff") {
    // A bitmap whose meaningful bits are all valid, with garbage (set) padding
    // bits in the partial final word — the realistic all-valid representation.
    const std::size_t nbits = 65;  // one full word + one bit
    std::vector<std::uint64_t> w(words(nbits));
    validity::fill_all_valid(w.data(), nbits);
    w.back() = ~0ull;  // every padding bit on

    const std::size_t ref = simd::count_set_scalar(w.data(), nbits);
    REQUIRE(ref == 65);

    // The REAL kernel matches the reference (suite PASSES on correct code).
    CHECK(simd::count_set_vec(w.data(), nbits) == ref);

    // The PLANTED MUTANT does not — it counts the padding bits. The exact
    // differential the suite uses flags it. THIS is the proof the check bites.
    const std::size_t buggy = simd::mutant::count_set_vec_tailbug(w.data(), nbits);
    CHECK(buggy != ref);       // <-- the mutation is detected
    CHECK(buggy == 65 + 63);   // counts all 64 bits of the last word
}

TEST_CASE("MUTATION: all_valid tail bug is CAUGHT (null in the partial word)") {
    const std::size_t nbits = 65;
    std::vector<std::uint64_t> w(words(nbits));
    validity::fill_all_valid(w.data(), nbits);
    validity::set_bit(w.data(), 64, false);  // null lives in the partial word

    // Reference + real kernel both correctly say "not all valid".
    REQUIRE(simd::all_valid_scalar(w.data(), nbits) == false);
    CHECK(simd::all_valid_vec(w.data(), nbits) == false);

    // The mutant SKIPS the partial word and wrongly reports all-valid. Caught.
    CHECK(simd::mutant::all_valid_vec_tailbug(w.data(), nbits) == true);
    CHECK(simd::mutant::all_valid_vec_tailbug(w.data(), nbits) !=
          simd::all_valid_scalar(w.data(), nbits));  // <-- detected
}

TEST_CASE("MUTATION: bug is DORMANT on word-aligned lengths (it is a tail bug)") {
    // On exact multiples of 64 there is no partial word, so the mutant equals the
    // real kernel — confirming the defect is specifically in tail handling, the
    // §12 hotspot, and that a suite testing ONLY aligned lengths would miss it.
    for (std::size_t nbits : {64u, 128u, 256u}) {
        std::vector<std::uint64_t> w(words(nbits));
        validity::fill_all_valid(w.data(), nbits);
        CHECK(simd::mutant::count_set_vec_tailbug(w.data(), nbits) ==
              simd::count_set_vec(w.data(), nbits));
        CHECK(simd::mutant::all_valid_vec_tailbug(w.data(), nbits) ==
              simd::all_valid_vec(w.data(), nbits));
    }
}

TEST_CASE("MUTATION: randomized sweep — the diff catches the mutant on many "
          "partial-word lengths, and never falsely accuses the real kernel") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xD1Fu);
    int caught = 0;
    for (int iter = 0; iter < 3000; ++iter) {
        std::size_t nbits = 1 + rng() % 4096;
        std::vector<std::uint64_t> w(words(nbits));
        for (auto& x : w) x = rng();  // garbage padding included

        const std::size_t ref = simd::count_set_scalar(w.data(), nbits);
        // The real kernel must ALWAYS match the reference.
        REQUIRE(simd::count_set_vec(w.data(), nbits) == ref);
        // Whenever there is a partial word with set padding, the mutant diverges.
        if (simd::mutant::count_set_vec_tailbug(w.data(), nbits) != ref) ++caught;
    }
    // Over thousands of random partial-word lengths the mutant is caught many
    // times — the differential is not a no-op.
    CHECK(caught > 100);
}
