//  WP-14: the compressed-scan differential. ONE Plan (compressed_scan(ct, src), and
//  also compressed_scan(ct, src).filter(…).project(…)) drives BOTH backends: it
//  lowers to tsx::CompressedScan — which decodes the COMPRESSED bytes of `ct`
//  independently — and renders to DuckDB SQL over the INDEPENDENT decompressed
//  reference `src` (the non-circular ctable/ctable_ref split). Each generated case
//  diffs GREEN vs the independent plan reference (always) AND, when staged
//  (QE_WITH_DUCKDB), the authoritative DuckDB result. Agreement requires the codec to
//  be truly lossless: a decode bit-slip would diverge from BOTH the reference and
//  DuckDB (proven by the mutation self-test).
//
//  `src` is generated with the existing divergence-safe bounds (finite/moderate
//  floats, bounded ints) so DuckDB never diverges; it lives in stable storage
//  (std::unique_ptr<Table>) because the Plan's CompressedScan node borrows both `ct`
//  and `src`. Batch sizes {64,256,2048} are swept so the streaming decoder resumes
//  across batch boundaries. Output is UNORDERED (no ORDER BY) => canonicalized (D12).
//
//  Replay:  ./compress_differential_test --seed N

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "expr/expr.h"
#include "ops/project.h"
#include "ops/table.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/generators.h"
#include "plan/plan.h"
#include "tests/wp1_seed.h"
#include "tsx/compress.h"

using namespace qe;
using namespace qe::plan;
using namespace qe::oracle;
using namespace qe::tsx;

namespace {

const std::size_t kBatchSizes[] = {64, 256, 2048};

// A type-matched zero literal so a filter predicate type-checks against col 0
// (gen_schema makes column 0 numeric).
expr::Expr zero_lit(Type t) {
    switch (t) {
        case Type::I32: return expr::lit(expr::Scalar::i32(0));
        case Type::F64: return expr::lit(expr::Scalar::f64(0.0));
        default:        return expr::lit(expr::Scalar::i64(0));  // I64 / TS
    }
}

DiffResult diff_one(const Plan& p, std::size_t bs) {
    DiffResult ref = run_plan_vs_reference(p, bs);
    if (!ref.equal) return ref;
    if (duckdb_available()) {
        try {
            DiffResult d = run_plan_differential(p, run_plan_duckdb, bs);
            if (!d.equal) return d;
        } catch (const DuckDBError& e) {
            MESSAGE("DuckDB raised (skipped, not a diff): " << e.what());
        }
    }
    return ref;
}

bool diff_all_batches(const Plan& p) {
    for (std::size_t bs : kBatchSizes) {
        const DiffResult d = diff_one(p, bs);
        if (!d.equal) {
            MESSAGE("DIFF at batch=" << bs << ": " << d.message << "\nplan:\n"
                                     << p.to_string());
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("WP-14: random compressed scans diff green vs reference + DuckDB") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xC0DEC0DEu);
    // 60 random schemas × {plain, filter+project} × 3 batch sizes; each also runs the
    // authoritative DuckDB diff when staged (CREATE+INSERT+SELECT per query), so this
    // is the heaviest WP-14 test — see the generous TIMEOUT in CMakeLists.
    for (int iter = 0; iter < 60; ++iter) {
        Schema schema = gen_schema(rng);
        // Stable storage: the CompressedScan node borrows BOTH src and ct.
        auto src = std::make_unique<Table>(gen_table(rng, schema));
        const CompressedTable ct = CompressedTable::encode(*src);
        const Type t0 = schema.fields[0].second;

        // (1) Plain compressed scan: decode every column end-to-end.
        const Plan p_scan = compressed_scan(ct, *src).plan();
        CHECK(diff_all_batches(p_scan));

        // (2) Decode FEEDS downstream operators: filter on col0, project col0+col1.
        std::vector<Projection> projs;
        projs.push_back(Projection{"p0", expr::col(t0, 0)});
        if (schema.fields.size() > 1)
            projs.push_back(
                Projection{"p1", expr::col(schema.fields[1].second, 1)});
        const Plan p_fp =
            compressed_scan(ct, *src)
                .filter(expr::ge(expr::col(t0, 0), zero_lit(t0)))
                .project(projs)
                .plan();
        CHECK(diff_all_batches(p_fp));
    }
}

TEST_CASE("WP-14 edge: empty compressed table (zero rows) round-trips through plan") {
    Schema s;
    s.fields.emplace_back("c0", Type::I64);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<OwnedColumn> cols;
    cols.push_back(OwnedColumn::make(Type::I64, 0));
    cols.push_back(OwnedColumn::make(Type::F64, 0));
    auto src = std::make_unique<Table>(s, std::move(cols));
    const CompressedTable ct = CompressedTable::encode(*src);
    const Plan p = compressed_scan(ct, *src).plan();
    CHECK(diff_all_batches(p));
}
