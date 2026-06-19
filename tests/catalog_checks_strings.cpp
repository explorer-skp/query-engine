//  WP-7b catalog check — operator orchestration: dictionary-encoded VARCHAR join
//  key wiring. This TU brings qe::mutant::StringKeyMutation + qe::mutant::
//  StringKeyJoin (ops/string_key_mutants.h) — a DISTINCT-named enum from the
//  join/sort/aggregate/asof/window Mutation enums, so it lives alone per the
//  conflict-free-TU convention.
//
//  The decisive comparison reproduces the §5 "by value, not by code" hazard: the
//  build and probe sides intern the SAME strings into INDEPENDENT dictionaries, so
//  the same value carries DIFFERENT int32 codes on the two sides. The kHashRawCode
//  mutant hashes/compares those raw codes, missing every cross-dictionary match —
//  flagged by the differential (independent reference + DuckDB VARCHAR join) while
//  the real operator passes.

#include "tests/catalog_checks.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/owned_batch.h"
#include "core/string_dict.h"
#include "core/types.h"
#include "ops/scan.h"
#include "ops/string_key_mutants.h"
#include "ops/table.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"

#include "tests/catalog_check_util.h"

namespace qe::catalog::checks {

using namespace qe;
using namespace qe::oracle;

namespace {
OwnedColumn str_col(const std::vector<std::string>& vals) {
    auto dict = std::make_shared<StringDict>();
    OwnedColumn c = OwnedColumn::make_str(vals.size(), dict);
    auto* codes = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < vals.size(); ++i)
        codes[i] = dict->intern(vals[i]);
    return c;
}
}  // namespace

Verdict string_key_by_code() {
    // build: keys interned in forward order ("AAPL"=0, "MSFT"=1). probe: the SAME
    // values interned in a DIFFERENT order ("AAPL" becomes code 2 here), so a
    // by-code join misses; a by-value join matches.
    Schema bs;
    bs.fields.emplace_back("c0", Type::STR);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(str_col({"AAPL", "MSFT", "GOOG"}));
    bcols.push_back(i64_col({10, 20, 30}));
    const Table build(bs, std::move(bcols));

    Schema ps;
    ps.fields.emplace_back("c0", Type::STR);
    std::vector<OwnedColumn> pcols;
    pcols.push_back(str_col({"GOOG", "MSFT", "AAPL", "ZZZ"}));
    const Table probe(ps, std::move(pcols));

    const JoinQuery jq{{0}, {0}, JoinType::Inner};
    const ResultSet ref = run_join_reference(probe, build, jq);
    const auto duckdb = [&]() { return run_join_duckdb(probe, build, jq); };

    // Clean engine = the mutant's kNone control (identical to the real HashJoin).
    auto p1 = std::make_unique<Scan>(probe, 2048);
    auto b1 = std::make_unique<Scan>(build, 2048);
    mutant::StringKeyJoin real(std::move(p1), std::move(b1), jq.probe_keys,
                               jq.build_keys, jq.type,
                               mutant::StringKeyMutation::kNone);
    const ResultSet eng = drain_operator(real);

    // Mutant = hash/compare STR keys by raw code.
    auto p2 = std::make_unique<Scan>(probe, 2048);
    auto b2 = std::make_unique<Scan>(build, 2048);
    mutant::StringKeyJoin mut(std::move(p2), std::move(b2), jq.probe_keys,
                              jq.build_keys, jq.type,
                              mutant::StringKeyMutation::kHashRawCode);
    const ResultSet meng = drain_operator(mut);

    Verdict v;
    v.clean_passes = diff_with_duckdb(eng, ref, duckdb, /*ordered=*/false).equal;
    const DiffResult md = diff_with_duckdb(meng, ref, duckdb, /*ordered=*/false);
    v.mutant_flagged = !md.equal;
    v.detail = md.message;
    return v;
}

}  // namespace qe::catalog::checks
