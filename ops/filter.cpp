//  WP-3: Filter implementation. See ops/filter.h.
#include "ops/filter.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/selection.h"
#include "core/types.h"
#include "core/validity.h"

namespace qe {

Filter::Filter(std::unique_ptr<Operator> child, expr::Expr predicate)
    : child_(std::move(child)), predicate_(std::move(predicate)) {}

void Filter::open() {
    if (predicate_.type() != Type::BOOL)
        throw std::invalid_argument("Filter: predicate must be BOOL");
    child_->open();
}

void Filter::close() { child_->close(); }

Schema Filter::output_schema() const { return child_->output_schema(); }

std::optional<Batch> Filter::next() {
    while (true) {
        std::optional<Batch> in = child_->next();
        if (!in) return std::nullopt;
        const Batch& b = *in;

        // Defensive: never hand a 0-row batch to expr::evaluate (it aborts on
        // an empty batch — see WP-3 report). Our Scan never emits one, but a
        // future child operator might; treat it as "nothing here, pull again".
        if (b.row_count == 0) continue;

        // evaluate() applies b.sel and returns a dense BOOL column of row_count.
        OwnedColumn pred = expr::evaluate(predicate_, b);
        const Column pv = pred.view();
        const auto* bits = reinterpret_cast<const std::uint8_t*>(pv.data);

        // Collect the PHYSICAL row indices that pass. Mapping logical row k of
        // the (possibly selected) input to its physical row is sel_at(b.sel, k);
        // compacting through these indices yields a dense result and dodges the
        // §12 aliasing hazard (compact_column is out-of-place).
        std::vector<std::uint32_t> pass;
        pass.reserve(b.row_count);
        for (std::size_t k = 0; k < b.row_count; ++k) {
            const bool valid =
                pv.all_valid || validity::get_bit(pv.validity, k);
            if (valid && bits[k] != 0) pass.push_back(sel_at(b.sel, k));
        }

        if (pass.empty()) continue;  // never emit an empty batch; pull again

        const SelectionVector psel{pass.data(), pass.size()};
        OwnedBatch out;
        for (const Column& col : b.cols)
            out.add_column(compact_column(col, &psel, pass.size()));

        current_ = std::move(out);
        return current_.view();
    }
}

}  // namespace qe
