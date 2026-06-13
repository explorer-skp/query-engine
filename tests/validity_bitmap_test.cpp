//  WP-1: validity-bitmap kernel tests. Establishes three things the engine's
//  credibility rests on before the DuckDB oracle exists (wired at WP-3):
//   1. scalar == vector for every bitmap kernel, over randomized inputs (seeded);
//   2. both agree with an independent per-bit brute-force reference;
//   3. all the boundary/edge cases enumerated in the WP-1 brief.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/validity.h"
#include "simd/validity_kernels.h"
#include "tests/wp1_seed.h"

using namespace qe;
using qe::validity::words;

namespace {

// Independent reference: count meaningful set bits one at a time (no popcount,
// no masking tricks) — a third implementation to triangulate scalar vs vector.
std::size_t brute_count(const std::uint64_t* w, std::size_t nbits) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < nbits; ++i) {
        if ((w[i >> 6] >> (i & 63u)) & 1ull) ++c;
    }
    return c;
}
bool brute_all_valid(const std::uint64_t* w, std::size_t nbits) {
    for (std::size_t i = 0; i < nbits; ++i) {
        if (!((w[i >> 6] >> (i & 63u)) & 1ull)) return false;
    }
    return true;
}

// Random words with DELIBERATELY GARBAGE padding bits in the final word, so a
// kernel that forgets to mask is exposed.
std::vector<std::uint64_t> random_bitmap(std::mt19937_64& rng, std::size_t nbits) {
    std::vector<std::uint64_t> w(words(nbits) == 0 ? 1 : words(nbits));
    for (auto& x : w) x = rng();
    return w;
}

}  // namespace

TEST_CASE("count_set / all_valid: scalar == vector == brute over random inputs") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x511u);
    for (int iter = 0; iter < 4000; ++iter) {
        const std::size_t nbits = rng() % 5000;  // spans many partial words
        auto w = random_bitmap(rng, nbits);

        const std::size_t cs = simd::count_set_scalar(w.data(), nbits);
        const std::size_t cv = simd::count_set_vec(w.data(), nbits);
        const std::size_t cb = brute_count(w.data(), nbits);
        CHECK(cs == cb);
        CHECK(cv == cb);

        const bool av_s = simd::all_valid_scalar(w.data(), nbits);
        const bool av_v = simd::all_valid_vec(w.data(), nbits);
        const bool av_b = brute_all_valid(w.data(), nbits);
        CHECK(av_s == av_b);
        CHECK(av_v == av_b);

        CHECK(simd::any_null_scalar(w.data(), nbits) == !av_b);
        CHECK(simd::any_null_vec(w.data(), nbits) == !av_b);
    }
}

TEST_CASE("bit_and / bit_or: scalar == vector, and encode null propagation") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xA17u);
    for (int iter = 0; iter < 3000; ++iter) {
        const std::size_t nbits = 1 + rng() % 4096;
        auto a = random_bitmap(rng, nbits);
        auto b = random_bitmap(rng, nbits);
        const std::size_t n = words(nbits);
        std::vector<std::uint64_t> os(n), ov(n);

        simd::bit_and_scalar(a.data(), b.data(), os.data(), nbits);
        simd::bit_and_vec(a.data(), b.data(), ov.data(), nbits);
        CHECK(os == ov);
        // AND is the null-propagation primitive: valid iff both inputs valid.
        for (std::size_t i = 0; i < nbits; ++i) {
            const bool ea = (a[i >> 6] >> (i & 63u)) & 1ull;
            const bool eb = (b[i >> 6] >> (i & 63u)) & 1ull;
            const bool er = (os[i >> 6] >> (i & 63u)) & 1ull;
            CHECK(er == (ea && eb));
        }

        simd::bit_or_scalar(a.data(), b.data(), os.data(), nbits);
        simd::bit_or_vec(a.data(), b.data(), ov.data(), nbits);
        CHECK(os == ov);
    }
}

TEST_CASE("edge: all-null and none-null (all_valid fast path)") {
    for (std::size_t nbits : {1u, 7u, 63u, 64u, 65u, 128u, 200u, 2048u}) {
        std::vector<std::uint64_t> w(words(nbits));

        validity::fill_all_valid(w.data(), nbits);
        CHECK(simd::count_set_scalar(w.data(), nbits) == nbits);
        CHECK(simd::count_set_vec(w.data(), nbits) == nbits);
        CHECK(simd::all_valid_scalar(w.data(), nbits));
        CHECK(simd::all_valid_vec(w.data(), nbits));
        CHECK_FALSE(simd::any_null_vec(w.data(), nbits));

        validity::fill_all_null(w.data(), nbits);
        CHECK(simd::count_set_scalar(w.data(), nbits) == 0);
        CHECK(simd::count_set_vec(w.data(), nbits) == 0);
        CHECK_FALSE(simd::all_valid_scalar(w.data(), nbits));
        CHECK_FALSE(simd::all_valid_vec(w.data(), nbits));
        CHECK(simd::any_null_vec(w.data(), nbits));
    }
}

TEST_CASE("edge: single bit, exact word boundaries, partial last word") {
    // Word boundaries and just-around: 63/64/65 and multiples of 64.
    for (std::size_t nbits : {1u, 63u, 64u, 65u, 127u, 128u, 129u, 191u, 192u,
                              193u, 256u, 257u}) {
        std::vector<std::uint64_t> w(words(nbits), 0ull);

        // Single bit set at the very last position (in the partial word when
        // nbits%64 != 0). Both kernels must see exactly one valid value.
        validity::set_bit(w.data(), nbits - 1, true);
        CHECK(simd::count_set_scalar(w.data(), nbits) == 1);
        CHECK(simd::count_set_vec(w.data(), nbits) == 1);
        // Exactly one valid bit => all-valid only in the degenerate nbits==1 case.
        CHECK(simd::all_valid_vec(w.data(), nbits) == (nbits == 1));

        // The CRITICAL partial-word case: all meaningful bits valid, but the
        // padding bits beyond nbits are GARBAGE (set to 1). A correct kernel
        // masks them; a tail-buggy one would over-count / wrongly accept.
        validity::fill_all_valid(w.data(), nbits);
        w.back() = ~0ull;  // force every padding bit on
        CHECK(simd::count_set_scalar(w.data(), nbits) == nbits);
        CHECK(simd::count_set_vec(w.data(), nbits) == nbits);
        CHECK(simd::all_valid_scalar(w.data(), nbits));
        CHECK(simd::all_valid_vec(w.data(), nbits));

        // And a null hiding in the partial last word must be detected.
        if (nbits >= 1) {
            validity::set_bit(w.data(), nbits - 1, false);
            CHECK_FALSE(simd::all_valid_scalar(w.data(), nbits));
            CHECK_FALSE(simd::all_valid_vec(w.data(), nbits));
            CHECK(simd::count_set_vec(w.data(), nbits) == nbits - 1);
        }
    }
}

TEST_CASE("edge: nbits == 0 is well defined") {
    std::uint64_t w = ~0ull;
    CHECK(simd::count_set_scalar(&w, 0) == 0);
    CHECK(simd::count_set_vec(&w, 0) == 0);
    CHECK(simd::all_valid_scalar(&w, 0));  // vacuously true
    CHECK(simd::all_valid_vec(&w, 0));
}
