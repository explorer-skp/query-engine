//  WP-3: Project implementation. See ops/project.h.
#include "ops/project.h"

#include <utility>

namespace qe {

Project::Project(std::unique_ptr<Operator> child,
                 std::vector<Projection> projections)
    : child_(std::move(child)), projections_(std::move(projections)) {}

void Project::open() { child_->open(); }

void Project::close() { child_->close(); }

Schema Project::output_schema() const {
    Schema s;
    s.fields.reserve(projections_.size());
    for (const auto& p : projections_)
        s.fields.emplace_back(p.name, p.expr.type());
    return s;
}

std::optional<Batch> Project::next() {
    std::optional<Batch> in = child_->next();
    if (!in) return std::nullopt;
    const Batch& b = *in;

    OwnedBatch out;
    if (b.row_count == 0) {
        // Defensive: expr::evaluate aborts on a 0-row batch (see WP-3 report).
        // Emit a typed empty batch instead of evaluating. Our children never
        // produce one, but this keeps Project total over the Operator contract.
        for (const auto& p : projections_)
            out.add_column(OwnedColumn::make(p.expr.type(), 0));
    } else {
        for (const auto& p : projections_)
            out.add_column(expr::evaluate(p.expr, b));
    }

    current_ = std::move(out);
    return current_.view();
}

}  // namespace qe
