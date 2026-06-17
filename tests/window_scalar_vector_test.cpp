//  WP-13: scalar==vector end-to-end check for windowed aggregation. The SLIDING
//  output's child-column MATERIALIZATION runs through the WP-1 selection gather
//  kernels (simd/gather_kernels.h) via the shared emitter (ops/join_internal.h
//  emit_build_column); GatherPath selects the Highway vector gather vs its
//  independently-written scalar twin. The TUMBLING partition step runs through the
//  frozen HashTable, whose HashPath selects the vector vs scalar key hash. Forcing
//  each path on the SAME generated input and asserting BYTE-IDENTICAL output (same
//  operator, same row order => positional compare) is the RIGOR.md rule-3 / D17
//  check that the vector kernel agrees with its scalar reference.
//
//  Applicability (per the WP brief): the bucket-scatter (tumbling) and the
//  ring-buffer cursor (sliding) are sequential scalar control flow (the WP-5/WP-6
//  precedent — no vector twin); the vectorized steps under test here are the sliding
//  child-column GATHER (GatherPath) and the partition key HASH (HashPath), exercised
//  on both paths.
//
//  Replay:  ./window_scalar_vector_test --seed N

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
#include "tsx/window.h"

using namespace qe;
using namespace qe::oracle;

namespace {

ResultSet run_paths(const WindowCase& c, std::size_t bs, HashPath hp,
                    GatherPath gp) {
    auto sc = std::make_unique<Scan>(c.input, bs);
    tsx::Window w(std::move(sc), c.mode, c.keys, c.time, c.param, c.aggs);
    w.set_paths(hp, gp);
    return drain_operator(w);
}

}  // namespace

TEST_CASE("WP-13: vector gather == scalar gather (and vector hash == scalar)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5C13D0u);
    for (int iter = 0; iter < 150; ++iter) {
        const WindowCase c = gen_window_case(rng);
        const std::size_t bs = (iter % 2) ? 256 : 2048;

        // Production path (vector hash + vector gather) is the baseline.
        const ResultSet base =
            run_paths(c, bs, HashPath::kVector, GatherPath::kVector);

        // Scalar gather twin must reproduce it byte-for-byte (same op => ORDERED).
        const ResultSet scalar_gather =
            run_paths(c, bs, HashPath::kVector, GatherPath::kScalar);
        DiffResult dg = compare_result_sets(base, scalar_gather, /*ordered=*/true);
        CHECK_MESSAGE(dg.equal, "gather scalar!=vector: " << dg.message);

        // Scalar hash twin too (the tumbling partition step).
        const ResultSet scalar_hash =
            run_paths(c, bs, HashPath::kScalar, GatherPath::kVector);
        DiffResult dh = compare_result_sets(base, scalar_hash, /*ordered=*/true);
        CHECK_MESSAGE(dh.equal, "hash scalar!=vector: " << dh.message);
    }
}
