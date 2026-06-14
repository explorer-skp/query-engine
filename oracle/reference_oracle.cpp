//  WP-3 (seed of WP-9): independent reference oracle. See reference_oracle.h.
#include "oracle/reference_oracle.h"

#include <cstdint>
#include <vector>

#include "core/validity.h"
#include "expr/expr.h"

namespace qe::oracle {
namespace {

// Read cell at dense index `r` from a dense (unselected) evaluated column.
Cell read_dense(const OwnedColumn& oc, std::size_t r) {
    const Column c = oc.view();
    Cell cell;
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) {
        cell.is_null = true;
        return cell;
    }
    switch (c.type) {
        case Type::I32:
            cell.i = reinterpret_cast<const std::int32_t*>(c.data)[r];
            break;
        case Type::I64:
        case Type::TS:
            cell.i = reinterpret_cast<const std::int64_t*>(c.data)[r];
            break;
        case Type::BOOL:
            cell.i = reinterpret_cast<const std::uint8_t*>(c.data)[r];
            break;
        case Type::F64:
            cell.f = reinterpret_cast<const double*>(c.data)[r];
            break;
    }
    return cell;
}

}  // namespace

ResultSet run_reference(const Table& table, const LogicalQuery& q) {
    const Batch full = table.full_batch_view();
    const std::size_t n = table.num_rows();

    // 0. Empty table: result is trivially empty. (Also a guard: expr::evaluate()
    //    -> compact_column aborts on a row_count==0 batch — see WP-3 report.
    //    Like the engine, which short-circuits a 0-row scan, we never evaluate.)
    if (n == 0) {
        ResultSet rs;
        for (const auto& p : q.projections) rs.types.push_back(p.expr.type());
        return rs;
    }

    // 1. Which rows pass the filter (default: all rows). NULL predicate => no.
    std::vector<char> keep(n, 1);
    if (q.has_filter()) {
        OwnedColumn pred = expr::evaluate(q.filter, full);
        const Column pv = pred.view();
        const auto* bits = reinterpret_cast<const std::uint8_t*>(pv.data);
        for (std::size_t r = 0; r < n; ++r) {
            const bool valid =
                pv.all_valid || validity::get_bit(pv.validity, r);
            keep[r] = (valid && bits[r] != 0) ? 1 : 0;
        }
    }

    // 2. Evaluate each projection over the whole table (dense, length n).
    ResultSet rs;
    std::vector<OwnedColumn> proj;
    proj.reserve(q.projections.size());
    for (const auto& p : q.projections) {
        proj.push_back(expr::evaluate(p.expr, full));
        rs.types.push_back(p.expr.type());
    }

    // 3. Emit one ResultSet row per kept table row.
    for (std::size_t r = 0; r < n; ++r) {
        if (!keep[r]) continue;
        std::vector<Cell> row;
        row.reserve(proj.size());
        for (const auto& pc : proj) row.push_back(read_dense(pc, r));
        rs.rows.push_back(std::move(row));
    }
    return rs;
}

}  // namespace qe::oracle
