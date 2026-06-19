//  WP-7b MUTATION SELF-TEST (mandatory; RIGOR.md rule 4: "a checker that cannot
//  fail proves nothing"). We plant a deliberately-broken STRING-KEY join
//  (ops/string_key_mutants.{h,cpp}) — a faithful copy of the real HashJoin reusing
//  the SAME ops/join_internal.h plumbing, whose ONLY defect is hashing/comparing
//  STR join keys by RAW int32 CODE instead of by string VALUE — and SHOW the
//  differential (independent reference + DuckDB VARCHAR when staged) CATCHES it,
//  while the REAL operator (and the mutant's kNone control) PASS the identical diff.
//
//  Why it bites: build and probe carry INDEPENDENT dictionaries, so the same string
//  gets DIFFERENT codes on the two sides; comparing codes misses every
//  cross-dictionary match. This is exactly the §5 "by value, not by code" hazard.
//
//  Replay:  ./wp7b_strings_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"

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
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::oracle;

namespace {

OwnedColumn str_col(const std::vector<std::optional<std::string>>& vals) {
    auto dict = std::make_shared<StringDict>();
    OwnedColumn c = OwnedColumn::make_str(vals.size(), dict);
    auto* codes = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        if (vals[i])
            codes[i] = dict->intern(*vals[i]);
        else
            c.set_null(i);
    }
    return c;
}

OwnedColumn i64_col(const std::vector<std::int64_t>& v) {
    OwnedColumn c = OwnedColumn::make(Type::I64, v.size());
    auto* d = reinterpret_cast<std::int64_t*>(c.mutable_data());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return c;
}

struct Case {
    Table probe;
    Table build;
    JoinQuery query;
};

// A cross-dictionary case: the same strings, interned in DIFFERENT orders on the
// two sides (so equal values carry different codes), with real matches.
Case cross_dict_case() {
    Schema bs;
    bs.fields.emplace_back("c0", Type::STR);
    bs.fields.emplace_back("c1", Type::I64);
    std::vector<OwnedColumn> bcols;
    bcols.push_back(str_col({"AAPL", "MSFT", "GOOG"}));  // codes 0,1,2
    bcols.push_back(i64_col({10, 20, 30}));
    Schema ps;
    ps.fields.emplace_back("c0", Type::STR);
    std::vector<OwnedColumn> pcols;
    // interned in REVERSE first-use order => "AAPL" is code 2 here, not 0.
    pcols.push_back(str_col({"GOOG", "MSFT", "AAPL", "NONE"}));
    JoinQuery q{{0}, {0}, JoinType::Inner};
    return {Table(ps, std::move(pcols)), Table(bs, std::move(bcols)),
            std::move(q)};
}

DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_join_reference(c.probe, c.build, c.query);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            d = compare_result_sets(engine,
                                    run_join_duckdb(c.probe, c.build, c.query));
        } catch (const DuckDBError&) { /* divergence backstop */ }
    }
    return d;
}

DiffResult run_string_mutant(const Case& c, mutant::StringKeyMutation m,
                             std::size_t bs) {
    auto ps = std::make_unique<Scan>(c.probe, bs);
    auto bsn = std::make_unique<Scan>(c.build, bs);
    mutant::StringKeyJoin j(std::move(ps), std::move(bsn), c.query.probe_keys,
                            c.query.build_keys, c.query.type, m);
    return diff_tree_vs_oracles(j, c);
}

}  // namespace

TEST_CASE("WP-7b MUTATION: the REAL string join passes the differential") {
    const Case c = cross_dict_case();
    for (std::size_t bs : {2u * 64u, 2048u}) {
        // the actual engine operator
        auto tree = build_join_pipeline(c.probe, c.build, c.query, bs);
        CHECK(diff_tree_vs_oracles(*tree, c).equal);
        // the mutant's kNone control == real behavior
        CHECK(run_string_mutant(c, mutant::StringKeyMutation::kNone, bs).equal);
    }
}

TEST_CASE("WP-7b MUTATION: kHashRawCode (compare STR keys by code) is CAUGHT") {
    const Case c = cross_dict_case();
    const DiffResult d =
        run_string_mutant(c, mutant::StringKeyMutation::kHashRawCode, 2048);
    CHECK_FALSE(d.equal);  // non-vacuous: the differential MUST flag it
    MESSAGE("kHashRawCode caught: " << d.message);
}
