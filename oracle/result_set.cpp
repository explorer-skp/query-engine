//  WP-3 (seed of WP-9): ResultSet drain + comparison. See oracle/result_set.h.
#include "oracle/result_set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>

#include "core/selection.h"
#include "core/validity.h"

namespace qe::oracle {
namespace {

// Read logical row `k` of column `c` (a batch column under selection `sel`).
Cell read_cell(const Column& c, const SelectionVector* sel, std::size_t k) {
    const std::uint32_t p = sel_at(sel, k);
    Cell cell;
    const bool valid = c.all_valid || validity::get_bit(c.validity, p);
    if (!valid) {
        cell.is_null = true;
        return cell;
    }
    switch (c.type) {
        case Type::I32:
            cell.i = reinterpret_cast<const std::int32_t*>(c.data)[p];
            break;
        case Type::I64:
        case Type::TS:
            cell.i = reinterpret_cast<const std::int64_t*>(c.data)[p];
            break;
        case Type::BOOL:
            cell.i = reinterpret_cast<const std::uint8_t*>(c.data)[p];
            break;
        case Type::F64:
            cell.f = reinterpret_cast<const double*>(c.data)[p];
            break;
    }
    return cell;
}

// A total order over cells of one type, for canonicalization (D12). NULLs sort
// first; then by value. Returns <0, 0, >0.
int cell_order(const Cell& a, const Cell& b, Type t) {
    if (a.is_null != b.is_null) return a.is_null ? -1 : 1;
    if (a.is_null) return 0;
    if (t == Type::F64) {
        if (a.f < b.f) return -1;
        if (a.f > b.f) return 1;
        return 0;  // equal or both NaN (NaN excluded by WP-3 generators)
    }
    if (a.i < b.i) return -1;
    if (a.i > b.i) return 1;
    return 0;
}

void canonicalize(ResultSet& rs) {
    std::sort(rs.rows.begin(), rs.rows.end(),
              [&](const std::vector<Cell>& x, const std::vector<Cell>& y) {
                  for (std::size_t c = 0; c < rs.types.size(); ++c) {
                      const int o = cell_order(x[c], y[c], rs.types[c]);
                      if (o != 0) return o < 0;
                  }
                  return false;
              });
}

bool float_eq(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    const double diff = std::fabs(a - b);
    return diff <= kAbsEps + kRelEps * std::max(std::fabs(a), std::fabs(b));
}

std::string cell_str(const Cell& c, Type t) {
    if (c.is_null) return "NULL";
    std::ostringstream os;
    if (t == Type::F64)
        os << c.f;
    else if (t == Type::BOOL)
        os << (c.i ? "true" : "false");
    else
        os << c.i;
    return os.str();
}

}  // namespace

ResultSet drain_operator(Operator& root) {
    ResultSet rs;
    const Schema schema = root.output_schema();
    for (const auto& f : schema.fields) rs.types.push_back(f.second);

    root.open();
    while (std::optional<Batch> b = root.next()) {
        const Batch& batch = *b;
        for (std::size_t k = 0; k < batch.row_count; ++k) {
            std::vector<Cell> row;
            row.reserve(batch.cols.size());
            for (const Column& col : batch.cols)
                row.push_back(read_cell(col, batch.sel, k));
            rs.rows.push_back(std::move(row));
        }
    }
    root.close();
    return rs;
}

DiffResult compare_result_sets(const ResultSet& engine_in,
                               const ResultSet& oracle_in, bool ordered) {
    DiffResult r;
    if (engine_in.types != oracle_in.types) {
        r.message = "column type/shape mismatch";
        return r;
    }
    if (engine_in.num_rows() != oracle_in.num_rows()) {
        std::ostringstream os;
        os << "row count mismatch: engine=" << engine_in.num_rows()
           << " oracle=" << oracle_in.num_rows();
        r.message = os.str();
        return r;
    }

    // UNORDERED (D12 default): canonicalize both sides so row order is irrelevant.
    // ORDERED (explicit ORDER BY, WP-7): compare positionally in emitted order,
    // NO canonicalization — that is what makes a wrong sort observable. The
    // cell-level rules (D11 epsilon / exact int / exact null) are identical.
    ResultSet e = engine_in, o = oracle_in;
    if (!ordered) {
        canonicalize(e);
        canonicalize(o);
    }
    const char* row_label = ordered ? "row" : "canonical row";

    for (std::size_t row = 0; row < e.num_rows(); ++row) {
        for (std::size_t c = 0; c < e.types.size(); ++c) {
            const Cell& ec = e.rows[row][c];
            const Cell& oc = o.rows[row][c];
            if (ec.is_null != oc.is_null) {
                std::ostringstream os;
                os << "null mismatch at " << row_label << " " << row << " col "
                   << c << ": engine=" << cell_str(ec, e.types[c])
                   << " oracle=" << cell_str(oc, o.types[c]);
                r.message = os.str();
                return r;
            }
            if (ec.is_null) continue;
            bool same;
            if (e.types[c] == Type::F64)
                same = float_eq(ec.f, oc.f);
            else
                same = ec.i == oc.i;
            if (!same) {
                std::ostringstream os;
                os << "value mismatch at " << row_label << " " << row << " col "
                   << c << ": engine=" << cell_str(ec, e.types[c])
                   << " oracle=" << cell_str(oc, o.types[c]);
                r.message = os.str();
                return r;
            }
        }
    }
    r.equal = true;
    return r;
}

std::string to_debug_string(const ResultSet& rs_in, std::size_t max_rows) {
    ResultSet rs = rs_in;
    canonicalize(rs);
    std::ostringstream os;
    os << rs.num_rows() << " rows x " << rs.num_cols() << " cols\n";
    const std::size_t cap = std::min(max_rows, rs.num_rows());
    for (std::size_t row = 0; row < cap; ++row) {
        os << "  [";
        for (std::size_t c = 0; c < rs.types.size(); ++c) {
            if (c) os << ", ";
            os << cell_str(rs.rows[row][c], rs.types[c]);
        }
        os << "]\n";
    }
    if (cap < rs.num_rows()) os << "  ... (" << (rs.num_rows() - cap) << " more)\n";
    return os.str();
}

}  // namespace qe::oracle
