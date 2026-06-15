//  WP-9: shared helpers for the mutation-catalog check TUs (tests/catalog_checks_*
//  .cpp). Header-only (inline) so each check TU links its own copy without an extra
//  object. These build small typed Tables and run the standard "engine vs
//  reference, escalating to DuckDB when staged" differential, returning a Verdict.
//
//  Why several check TUs at all: the per-WP mutant headers
//  (ops/{hashtable,aggregate,join}_mutants.h) each define a DISTINCT
//  qe::mutant::Mutation enum, so at most one of that trio may appear in any single
//  translation unit. The catalog therefore groups its checks into TUs that never
//  combine two conflicting headers; this header carries the parts they all share.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/operator.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "tests/mutation_catalog.h"

namespace qe::catalog::checks {

inline OwnedColumn i64_col(const std::vector<std::int64_t>& v,
                           const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

inline OwnedColumn i32_col(const std::vector<std::int32_t>& v,
                           const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::I32, v.size());
    auto* d = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

inline OwnedColumn f64_col(const std::vector<double>& v) {
    OwnedColumn c = OwnedColumn::make(Type::F64, v.size());
    auto* d = reinterpret_cast<double*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

// Diff `got` against the independent reference `ref`, escalating to the
// authoritative DuckDB result (via `duckdb`, evaluated lazily) only when DuckDB is
// staged and the reference already agreed — mirroring every per-WP differential.
inline qe::oracle::DiffResult diff_with_duckdb(
    const qe::oracle::ResultSet& got, const qe::oracle::ResultSet& ref,
    const std::function<qe::oracle::ResultSet()>& duckdb, bool ordered) {
    qe::oracle::DiffResult d = qe::oracle::compare_result_sets(got, ref, ordered);
    if (d.equal && qe::oracle::duckdb_available() && duckdb) {
        try {
            d = qe::oracle::compare_result_sets(got, duckdb(), ordered);
        } catch (const qe::oracle::DuckDBError&) { /* divergence backstop */ }
    }
    return d;
}

// Generic verdict for a SINGLE-TABLE LogicalQuery: the real engine pipeline must
// pass the differential (clean), and the caller-built `mutant_tree` must be flagged.
inline Verdict single_table_verdict(const Table& t,
                                    const qe::oracle::LogicalQuery& q,
                                    std::unique_ptr<Operator> mutant_tree,
                                    bool ordered) {
    using namespace qe::oracle;
    const ResultSet ref = run_reference(t, q);
    const auto duckdb = [&]() { return run_duckdb(t, q); };

    auto real = build_engine_pipeline(t, q);
    const ResultSet eng = drain_operator(*real);
    const ResultSet meng = drain_operator(*mutant_tree);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, ordered).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, ordered);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
