//  WP-3: Project — produce output columns by evaluating a list of expr trees
//  over each child batch. Each output column is expr::evaluate(expr, batch),
//  which applies the input selection vector and yields a dense OwnedColumn of
//  row_count values; Project bundles them into a fresh dense OwnedBatch. This is
//  where expression evaluation meets the oracle (Project = the SELECT list).
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "expr/expr.h"
#include "ops/operator.h"

namespace qe {

// One output column of a Project: an expression and the name it is published
// under (used by output_schema and, in the oracle, the SQL projection alias).
struct Projection {
    std::string name;
    expr::Expr expr;
};

class Project : public Operator {
   public:
    Project(std::unique_ptr<Operator> child, std::vector<Projection> projections);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    std::unique_ptr<Operator> child_;
    std::vector<Projection> projections_;
    OwnedBatch current_;  // backs the view returned by the last next()
};

}  // namespace qe
