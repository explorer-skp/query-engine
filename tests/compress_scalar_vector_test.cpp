//  WP-14: scalar==vector end-to-end check for the streaming decoder. The integer
//  codecs (TS/I64 delta-of-delta, I32 zig-zag) run a zig-zag-decode step that has a
//  Highway vector kernel (tsx/compress_kernels.cpp) and an independently-written
//  scalar twin (tsx/compress_scalar.cpp); CompressedScan's DecodePath seam selects
//  which. Forcing each path on the SAME compressed input and asserting BYTE-IDENTICAL
//  decoded output is the RIGOR.md rule-3 / D17 check that the vector kernel agrees
//  with its scalar reference.
//
//  Applicability (per the WP brief): the varint byte-extraction, the delta-of-delta
//  running prefix, and Gorilla reconstruction are inherently sequential and scalar on
//  BOTH paths; the vectorized step under test is the zig-zag transform, exercised
//  whenever an I32/I64/TS column is decoded. Replay: ./compress_scalar_vector_test --seed N

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/types.h"
#include "core/validity.h"
#include "ops/table.h"
#include "oracle/generators.h"
#include "tests/wp1_seed.h"
#include "tsx/compress.h"

using namespace qe;
using namespace qe::tsx;
using namespace qe::oracle;

namespace {

std::pair<bool, std::uint64_t> cell_bits(const Column& c, std::size_t r) {
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) return {false, 0};
    switch (c.type) {
        case Type::I32:
            return {true, static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                              reinterpret_cast<const std::int32_t*>(c.data)[r]))};
        case Type::I64:
        case Type::TS:
            return {true, static_cast<std::uint64_t>(
                              reinterpret_cast<const std::int64_t*>(c.data)[r])};
        case Type::F64: {
            std::uint64_t b;
            std::memcpy(&b, reinterpret_cast<const double*>(c.data) + r, 8);
            return {true, b};
        }
        case Type::BOOL:
            return {true, reinterpret_cast<const std::uint8_t*>(c.data)[r] ? 1u
                                                                           : 0u};
    }
    return {false, 0};
}

std::vector<std::vector<std::pair<bool, std::uint64_t>>> drain(
    const CompressedTable& ct, std::size_t bs, DecodePath path) {
    CompressedScan sc(ct, bs);
    sc.set_path(path);
    std::vector<std::vector<std::pair<bool, std::uint64_t>>> out(
        ct.num_columns());
    sc.open();
    while (auto b = sc.next()) {
        const Batch& batch = *b;
        for (std::size_t c = 0; c < ct.num_columns(); ++c)
            for (std::size_t r = 0; r < batch.row_count; ++r)
                out[c].push_back(cell_bits(batch.cols[c], r));
    }
    sc.close();
    return out;
}

}  // namespace

TEST_CASE("WP-14: vector zig-zag decode == scalar zig-zag decode (byte-identical)") {
    std::mt19937_64 rng(qe::test::seed() ^ 0x5CA1A5u);
    for (int iter = 0; iter < 100; ++iter) {
        Schema schema = gen_schema(rng);
        const Table src = gen_table(rng, schema);
        const CompressedTable ct = CompressedTable::encode(src);
        const std::size_t bs = (iter % 2) ? 256 : 2048;

        const auto vec = drain(ct, bs, DecodePath::kVector);
        const auto sca = drain(ct, bs, DecodePath::kScalar);
        REQUIRE(vec.size() == sca.size());
        for (std::size_t c = 0; c < vec.size(); ++c) {
            REQUIRE(vec[c].size() == sca[c].size());
            for (std::size_t r = 0; r < vec[c].size(); ++r)
                CHECK_MESSAGE(vec[c][r] == sca[c][r],
                              "path mismatch col=" << c << " row=" << r);
        }
    }
}
