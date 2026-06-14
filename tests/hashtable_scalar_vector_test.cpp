//  WP-4 GATE: scalar == vector for the group-find/probe path (RIGOR.md rule 3 /
//  D17). Two layers:
//    1. DIRECT kernel diff: hash_combine_vec == hash_combine_scalar over seeded
//       random words AND the SIMD-tail boundary lengths (the off-by-a-lane
//       hotspot). This is the differential that makes the mutation self-test of
//       the kernel meaningful.
//    2. END-TO-END: run a whole insert_or_find / find via HashPath::kVector vs
//       HashPath::kScalar and assert IDENTICAL group ids. Since the probe walk is
//       shared and only the hash kernel differs by path, equal ids prove the two
//       hash paths agree in situ, not just in isolation.
//  Replay any failure:  ./hashtable_scalar_vector_test --seed N

#include <cstdint>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "ops/hash_kernels.h"
#include "ops/hashtable.h"
#include "tests/hashtable_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::ht_test;

namespace {
const std::vector<std::size_t> kLens = {0,   1,   2,   7,    8,   15,  16,
                                        17,  31,  32,  33,   63,  64,  65,
                                        127, 128, 129, 1000, 2047, 2048};
}  // namespace

TEST_CASE("scalar==vector: hash_combine over random words, boundary lengths") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x4A5Eu);
    for (std::size_t n : kLens) {
        // Multi-column fold: re-run combine a few times with fresh words, exactly
        // as a composite key would, and compare the two paths step by step.
        std::vector<std::uint64_t> acc_v(n), acc_s(n);
        const std::uint64_t seed = rng();
        for (std::size_t k = 0; k < n; ++k) acc_v[k] = acc_s[k] = seed;
        for (int col = 0; col < 3; ++col) {
            std::vector<std::uint64_t> words(n);
            for (auto& w : words) w = rng();
            ops::hash_combine_vec(acc_v.data(), words.data(), n);
            ops::hash_combine_scalar(acc_s.data(), words.data(), n);
            CHECK(acc_v == acc_s);
        }
    }
}

TEST_CASE("end-to-end: insert_or_find ids identical for kVector and kScalar") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xE2Eu);
    for (int iter = 0; iter < 40; ++iter) {
        const std::size_t n = 1 + rng() % 1500;
        // Two key columns; small domain so there are real collisions and reuse.
        std::vector<std::int64_t> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<std::int64_t>(rng() % 64);
            b[i] = static_cast<std::int64_t>(rng() % 64);
        }
        KeyHolder hv, hs;
        hv.owned.push_back(make_col(Type::I32, a));
        hv.owned.push_back(make_col(Type::I64, b));
        hs.owned.push_back(make_col(Type::I32, a));
        hs.owned.push_back(make_col(Type::I64, b));

        HashTable tv({Type::I32, Type::I64});
        HashTable ts({Type::I32, Type::I64});
        std::vector<std::uint32_t> gv(n), gs(n);
        tv.insert_or_find(hv.kc(), n, gv.data(), HashPath::kVector);
        ts.insert_or_find(hs.kc(), n, gs.data(), HashPath::kScalar);
        CHECK(gv == gs);
        CHECK(tv.num_groups() == ts.num_groups());

        // And find() agrees across paths on a fresh probe batch.
        std::vector<std::int64_t> pa(n), pb(n);
        for (std::size_t i = 0; i < n; ++i) {
            pa[i] = static_cast<std::int64_t>(rng() % 80);
            pb[i] = static_cast<std::int64_t>(rng() % 80);
        }
        KeyHolder pv, ps;
        pv.owned.push_back(make_col(Type::I32, pa));
        pv.owned.push_back(make_col(Type::I64, pb));
        ps.owned.push_back(make_col(Type::I32, pa));
        ps.owned.push_back(make_col(Type::I64, pb));
        std::vector<std::uint32_t> fv(n), fs(n);
        tv.find(pv.kc(), n, fv.data(), HashPath::kVector);
        ts.find(ps.kc(), n, fs.data(), HashPath::kScalar);
        CHECK(fv == fs);
    }
}
