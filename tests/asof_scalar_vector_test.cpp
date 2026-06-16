//  WP-12: scalar==vector end-to-end check for the as-of join. The matched-build-row
//  (and carried-probe-row) MATERIALIZATION runs through the WP-1 selection gather
//  kernels (simd/gather_kernels.h) via the shared emitters (ops/join_internal.h);
//  GatherPath selects the Highway vector gather vs its independently-written scalar
//  twin. Forcing each path on the SAME generated input and asserting BYTE-IDENTICAL
//  output (same operator, same row order => positional compare) is the RIGOR.md
//  rule-3 / D17 check that the vector kernel agrees with its scalar reference.
//
//  Applicability (stated per the WP brief): the per-key MERGE itself is sequential
//  scalar control flow (no vector twin — the WP-6 probe-walk precedent); the
//  vectorized step under test here is the column GATHER, exercised on both paths.
//  HashPath is swept too (the HashTable's vector vs scalar key-hash) so the partition
//  step is also a true scalar==vector check.
//
//  Replay:  ./asof_scalar_vector_test --seed N

#include <cstddef>
#include <memory>

#include "doctest/doctest.h"

#include "ops/hashtable.h"  // HashPath
#include "ops/join.h"       // GatherPath
#include "ops/scan.h"
#include "ops/table.h"
#include "oracle/generators.h"
#include "oracle/result_set.h"
#include "tests/wp1_seed.h"
#include "tsx/asof.h"

using namespace qe;
using namespace qe::oracle;

namespace {

ResultSet run_paths(const AsofCase& c, std::size_t bs, HashPath hp,
                    GatherPath gp) {
    auto ps = std::make_unique<Scan>(c.probe, bs);
    auto bsn = std::make_unique<Scan>(c.build, bs);
    tsx::AsofJoin j(std::move(ps), std::move(bsn), c.left_keys, c.right_keys,
                    c.left_time, c.right_time, c.type, c.tolerance);
    j.set_paths(hp, gp);
    return drain_operator(j);
}

}  // namespace

TEST_CASE("WP-12: vector gather == scalar gather (and vector hash == scalar)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5CA1A5u);
    for (int iter = 0; iter < 150; ++iter) {
        const AsofCase c = gen_asof_case(rng);
        const std::size_t bs = (iter % 2) ? 256 : 2048;

        // Production path (vector hash + vector gather) is the baseline.
        const ResultSet base = run_paths(c, bs, HashPath::kVector,
                                         GatherPath::kVector);

        // Scalar gather twin must reproduce it byte-for-byte (same op => ORDERED).
        const ResultSet scalar_gather =
            run_paths(c, bs, HashPath::kVector, GatherPath::kScalar);
        DiffResult dg = compare_result_sets(base, scalar_gather, /*ordered=*/true);
        CHECK_MESSAGE(dg.equal, "gather scalar!=vector: " << dg.message);

        // Scalar hash twin too (the partition step).
        const ResultSet scalar_hash =
            run_paths(c, bs, HashPath::kScalar, GatherPath::kVector);
        DiffResult dh = compare_result_sets(base, scalar_hash, /*ordered=*/true);
        CHECK_MESSAGE(dh.equal, "hash scalar!=vector: " << dh.message);
    }
}
