//  WP-3 MUTATION SELF-TEST #1 (mandatory, RIGOR.md rule 4: "a checker that
//  cannot fail proves nothing"). We deliberately BREAK an operator and SHOW the
//  differential flags it; then the CORRECT operator passes the same diff.
//
//  Two planted mutants, both test-only (never linked into the engine):
//    * BuggyFilter — INVERTS the pass decision (keeps rows the predicate
//      rejects, incl. NULL-predicate rows). The classic "flipped filter".
//    * BuggyScan   — DROPS the final partial batch (the SIMD-tail / last-batch
//      omission hotspot from §12).
//
//  Each is run through the engine pipeline and diffed against the independent
//  reference oracle (and DuckDB when staged, QE_WITH_DUCKDB) — the IDENTICAL
//  comparator the green M1 test uses. The diff MUST come back not-equal.
//
//  Replay:  ./oracle_mutation_test --seed N

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "core/owned_batch.h"
#include "core/selection.h"
#include "core/types.h"
#include "core/validity.h"
#include "expr/expr.h"
#include "oracle/differential.h"
#include "oracle/duckdb_oracle.h"
#include "oracle/logical_query.h"
#include "oracle/reference_oracle.h"
#include "oracle/result_set.h"
#include "ops/filter.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/table.h"
#include "tests/ops_test_util.h"

using namespace qe;
using namespace qe::expr;
using namespace qe::oracle;
using namespace qe::ops_test;

namespace {

// MUTANT A: Filter that keeps exactly the rows the correct Filter drops.
class BuggyFilter : public Operator {
   public:
    BuggyFilter(std::unique_ptr<Operator> child, Expr predicate)
        : child_(std::move(child)), predicate_(std::move(predicate)) {}
    void open() override { child_->open(); }
    void close() override { child_->close(); }
    Schema output_schema() const override { return child_->output_schema(); }
    std::optional<Batch> next() override {
        while (true) {
            std::optional<Batch> in = child_->next();
            if (!in) return std::nullopt;
            const Batch& b = *in;
            OwnedColumn pred = evaluate(predicate_, b);
            const Column pv = pred.view();
            const auto* bits = reinterpret_cast<const std::uint8_t*>(pv.data);
            std::vector<std::uint32_t> pass;
            for (std::size_t k = 0; k < b.row_count; ++k) {
                const bool valid =
                    pv.all_valid || validity::get_bit(pv.validity, k);
                const bool correct = valid && bits[k] != 0;
                if (!correct) pass.push_back(sel_at(b.sel, k));  // INVERTED
            }
            if (pass.empty()) continue;
            const SelectionVector psel{pass.data(), pass.size()};
            OwnedBatch out;
            for (const Column& col : b.cols)
                out.add_column(compact_column(col, &psel, pass.size()));
            current_ = std::move(out);
            return current_.view();
        }
    }

   private:
    std::unique_ptr<Operator> child_;
    Expr predicate_;
    OwnedBatch current_;
};

// MUTANT B: Scan that silently omits the final partial batch.
class BuggyScan : public Operator {
   public:
    BuggyScan(const Table& t, std::size_t bs) : t_(t), bs_(bs) {}
    void open() override { cursor_ = 0; }
    void close() override {}
    Schema output_schema() const override { return t_.schema(); }
    std::optional<Batch> next() override {
        const std::size_t total = t_.num_rows();
        // BUG: only emit FULL batches; a trailing partial batch is dropped.
        if (cursor_ + bs_ > total) return std::nullopt;
        const std::size_t start = cursor_;
        Batch b;
        for (std::size_t c = 0; c < t_.num_columns(); ++c) {
            const OwnedColumn& oc = t_.column(c);
            Column v;
            v.type = oc.type();
            v.len = bs_;
            v.data = oc.data() + start * byte_width(oc.type());
            if (oc.all_valid()) {
                v.validity = nullptr;
                v.all_valid = true;
            } else {
                v.validity = oc.validity() + (start / 64);
                v.all_valid = false;
            }
            b.cols.push_back(v);
        }
        b.sel = nullptr;
        b.row_count = bs_;
        cursor_ = start + bs_;
        return b;
    }

   private:
    const Table& t_;
    std::size_t bs_;
    std::size_t cursor_ = 0;
};

// A representative table + query with a mid-selectivity predicate over a
// null-bearing column and a tail (200 is not a multiple of 64).
struct Case {
    Table table;
    LogicalQuery query;
};
Case make_case() {
    Schema s;
    s.fields.emplace_back("c0", Type::I32);
    s.fields.emplace_back("c1", Type::F64);
    std::vector<std::int32_t> a(200);
    std::vector<double> bvals(200);
    std::vector<std::size_t> nulls;
    for (std::size_t i = 0; i < 200; ++i) {
        a[i] = static_cast<std::int32_t>(i) - 100;  // spans + and -
        bvals[i] = static_cast<double>(i) * 0.25;
        if (i % 11 == 0) nulls.push_back(i);
    }
    std::vector<OwnedColumn> cols;
    cols.push_back(i32_col(a, nulls));
    cols.push_back(f64_col(bvals));
    LogicalQuery q;
    q.filter = gt(col(Type::I32, 0), lit(Scalar::i32(0)));  // ~half pass
    q.projections.push_back({"p0", col(Type::I32, 0)});
    q.projections.push_back({"p1", col(Type::F64, 1)});
    return Case{Table(s, std::move(cols)), std::move(q)};
}

// Diff a manually-built (possibly mutated) engine tree against the oracle(s).
DiffResult diff_tree_vs_oracles(Operator& tree, const Case& c) {
    const ResultSet engine = drain_operator(tree);
    const ResultSet ref = run_reference(c.table, c.query);
    DiffResult d = compare_result_sets(engine, ref);
    if (d.equal && duckdb_available()) {
        try {
            const ResultSet dk = run_duckdb(c.table, c.query);
            d = compare_result_sets(engine, dk);
        } catch (const DuckDBError& e) {
            // Grammar-safe case: a raise is a renderer/oracle regression, not a
            // divergence. Silently skipping would disable DuckDB coverage with
            // CI green (audit C3) -- fail loudly instead.
            FAIL("DuckDB raised on a grammar-safe case: " << std::string(e.what()));
        }
    }
    return d;
}

}  // namespace

TEST_CASE("mutation #1a: an INVERTED filter is flagged by the differential") {
    const Case c = make_case();

    // The CORRECT pipeline diffs green.
    CHECK(run_vs_reference(c.table, c.query, 64).equal);

    // The MUTATED pipeline (inverted filter) must be FLAGGED (not equal).
    auto scan = std::make_unique<Scan>(c.table, 64);
    auto bf = std::make_unique<BuggyFilter>(std::move(scan), c.query.filter);
    auto proj = std::make_unique<Project>(std::move(bf), c.query.projections);
    const DiffResult d = diff_tree_vs_oracles(*proj, c);
    CHECK_FALSE(d.equal);
    MESSAGE("inverted-filter mutant caught: " << d.message);
}

TEST_CASE("mutation #1b: a dropped final partial batch is flagged") {
    const Case c = make_case();  // 200 rows, batch 64 -> tail of 8 rows

    CHECK(run_vs_reference(c.table, c.query, 64).equal);

    auto scan = std::make_unique<BuggyScan>(c.table, 64);
    auto filt = std::make_unique<Filter>(std::move(scan), c.query.filter);
    auto proj = std::make_unique<Project>(std::move(filt), c.query.projections);
    const DiffResult d = diff_tree_vs_oracles(*proj, c);
    CHECK_FALSE(d.equal);
    MESSAGE("dropped-tail mutant caught: " << d.message);
}
