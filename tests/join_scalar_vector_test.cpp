//  WP-6 scalar==vector gate (RIGOR.md rule 3 / D17). Two levels:
//
//   1. DIRECT: the build-row gather reuses the WP-1 selection-vector gather
//      kernels (simd/gather_kernels.h). Here we drive gather32/gather64 through
//      BOTH the Highway vector path (_vec) and its independently-written scalar
//      twin (_scalar) on random (src, idx) and assert byte-identical output — the
//      same kernels the join's emit path calls.
//
//   2. END-TO-END: the join itself is run through every combination of
//      HashPath::{kVector,kScalar} (the probe key-hash path) and
//      GatherPath::{kVector,kScalar} (the build-column gather path), and all four
//      result sets must be identical (and equal to the independent reference). So
//      "scalar == vector" is a true end-to-end check of the operator, not just of
//      the kernels in isolation.
//
//  Replay:  ./join_scalar_vector_test --seed N

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/join.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/generators.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "simd/gather_kernels.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::oracle;

namespace {

ResultSet run_join_paths(const Table& p, const Table& b, const JoinQuery& q,
                         std::size_t bs, HashPath hp, GatherPath gp) {
    auto ps = std::make_unique<Scan>(p, bs);
    auto bsn = std::make_unique<Scan>(b, bs);
    HashJoin j(std::move(ps), std::move(bsn), q.probe_keys, q.build_keys, q.type);
    j.set_paths(hp, gp);
    return drain_operator(j);
}

}  // namespace

TEST_CASE("DIRECT: gather _vec == _scalar (the build-row gather kernels)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x6A33Eu);
    for (int trial = 0; trial < 64; ++trial) {
        const std::size_t srcN = 1 + rng() % 4096;
        const std::size_t n = 1 + rng() % 4096;
        std::vector<std::uint32_t> idx(n);
        for (auto& x : idx) x = static_cast<std::uint32_t>(rng() % srcN);

        // 32-bit lanes (I32 columns).
        std::vector<std::uint32_t> s32(srcN);
        for (auto& x : s32) x = static_cast<std::uint32_t>(rng());
        std::vector<std::uint32_t> o_vec(n), o_scl(n);
        simd::gather32_vec(s32.data(), idx.data(), n, o_vec.data());
        simd::gather32_scalar(s32.data(), idx.data(), n, o_scl.data());
        CHECK(o_vec == o_scl);

        // 64-bit lanes (I64/F64/TS columns).
        std::vector<std::uint64_t> s64(srcN);
        for (auto& x : s64) x = (static_cast<std::uint64_t>(rng()) << 32) ^ rng();
        std::vector<std::uint64_t> p_vec(n), p_scl(n);
        simd::gather64_vec(s64.data(), idx.data(), n, p_vec.data());
        simd::gather64_scalar(s64.data(), idx.data(), n, p_scl.data());
        CHECK(p_vec == p_scl);
    }
}

TEST_CASE("END-TO-END: join is identical across HashPath x GatherPath") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x10E2Du);

    for (int iter = 0; iter < 60; ++iter) {
        JoinCase jc = gen_join_case(rng);
        const std::size_t bs = (iter % 2) ? 64 : 256;

        const ResultSet base = run_join_paths(jc.probe, jc.build, jc.query, bs,
                                              HashPath::kVector,
                                              GatherPath::kVector);
        // The independent reference pins that the baseline is itself correct.
        const ResultSet ref = run_join_reference(jc.probe, jc.build, jc.query);
        CHECK(compare_result_sets(base, ref).equal);

        for (HashPath hp : {HashPath::kVector, HashPath::kScalar}) {
            for (GatherPath gp : {GatherPath::kVector, GatherPath::kScalar}) {
                const ResultSet rs =
                    run_join_paths(jc.probe, jc.build, jc.query, bs, hp, gp);
                const DiffResult d = compare_result_sets(base, rs);
                CHECK(d.equal);
                if (!d.equal) {
                    MESSAGE("path mismatch iter=" << iter
                                                  << " hp=" << (int)hp
                                                  << " gp=" << (int)gp << " : "
                                                  << d.message);
                }
            }
        }
    }
}
