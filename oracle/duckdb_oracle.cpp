//  WP-3 (seed of WP-9): DuckDB differential backend. See oracle/duckdb_oracle.h.
//
//  This translation unit is compiled ONLY when the DuckDB amalgamation is staged
//  (CMake sets QE_WITH_DUCKDB and adds third_party/duckdb to the include/link of
//  the oracle test target). It uses only the stable Connection::Query() +
//  MaterializedQueryResult + Value surface so it builds across DuckDB point
//  releases.
#include "oracle/duckdb_oracle.h"

#ifdef QE_WITH_DUCKDB

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "core/column.h"
#include "core/string_dict.h"
#include "core/validity.h"
#include "duckdb.hpp"
#include "oracle/plan_sql.h"
#include "oracle/sql_render.h"

namespace qe::oracle {
namespace {

constexpr const char* kTable = "t";

// Render one cell of `oc` at physical row r to a SQL literal for INSERT.
std::string cell_literal(const OwnedColumn& oc, std::size_t r) {
    const Column c = oc.view();
    const bool valid = c.all_valid || validity::get_bit(c.validity, r);
    if (!valid) return "NULL";
    std::ostringstream os;
    switch (c.type) {
        case Type::I32:
            os << reinterpret_cast<const std::int32_t*>(c.data)[r];
            break;
        case Type::I64:
        case Type::TS:
            os << reinterpret_cast<const std::int64_t*>(c.data)[r];
            break;
        case Type::BOOL:
            os << (reinterpret_cast<const std::uint8_t*>(c.data)[r] ? "TRUE"
                                                                    : "FALSE");
            break;
        case Type::F64:
            os.precision(17);
            os << reinterpret_cast<const double*>(c.data)[r];
            break;
        case Type::STR: {
            // WP-7b: decode the dict code to its bytes and emit a single-quoted SQL
            // string literal (doubling embedded quotes) so DuckDB sees the real
            // TEXT, not the code. Generators emit ASCII without embedded NULs.
            const auto code = reinterpret_cast<const std::int32_t*>(c.data)[r];
            const std::string_view sv = c.dict->at(code);
            os << '\'';
            for (const char ch : sv) {
                if (ch == '\'') os << "''";
                else os << ch;
            }
            os << '\'';
            break;
        }
    }
    return os.str();
}

void must(duckdb::Connection& con, const std::string& sql) {
    auto r = con.Query(sql);
    if (r->HasError()) throw DuckDBError(r->GetError() + "  [sql: " + sql + "]");
}

void load_named_table(duckdb::Connection& con, const std::string& name,
                      const Table& table) {
    must(con, create_table_sql(name, table.schema()));
    const std::size_t n = table.num_rows();
    const std::size_t ncol = table.num_columns();
    if (n == 0) return;

    // INSERT in chunks to keep each statement a reasonable size.
    constexpr std::size_t kChunk = 1000;
    for (std::size_t base = 0; base < n; base += kChunk) {
        const std::size_t end = std::min(base + kChunk, n);
        std::ostringstream os;
        os << "INSERT INTO " << name << " VALUES ";
        for (std::size_t r = base; r < end; ++r) {
            if (r != base) os << ", ";
            os << "(";
            for (std::size_t c = 0; c < ncol; ++c) {
                if (c) os << ", ";
                os << cell_literal(table.column(c), r);
            }
            os << ")";
        }
        must(con, os.str());
    }
}

void load_table(duckdb::Connection& con, const Table& table) {
    load_named_table(con, kTable, table);
}

Cell read_value(const duckdb::Value& v, Type t) {
    Cell cell;
    if (v.IsNull()) {
        cell.is_null = true;
        return cell;
    }
    switch (t) {
        case Type::I32:
            cell.i = v.GetValue<std::int32_t>();
            break;
        case Type::I64:
        case Type::TS:
            cell.i = v.GetValue<std::int64_t>();
            break;
        case Type::BOOL:
            cell.i = v.GetValue<bool>() ? 1 : 0;
            break;
        case Type::F64:
            cell.f = v.GetValue<double>();
            break;
        case Type::STR:
            cell.s = v.GetValue<std::string>();  // WP-7b: VARCHAR text by value
            break;
    }
    return cell;
}

// Read a rendered SQL result back into a ResultSet with the given column types
// (the single-source-of-truth output schema), in column order.
ResultSet read_result(duckdb::MaterializedQueryResult& result,
                      std::vector<Type> types) {
    ResultSet rs;
    rs.types = std::move(types);
    const std::size_t nrow = result.RowCount();
    const std::size_t ncol = rs.types.size();
    for (std::size_t r = 0; r < nrow; ++r) {
        std::vector<Cell> row;
        row.reserve(ncol);
        for (std::size_t c = 0; c < ncol; ++c)
            row.push_back(read_value(result.GetValue(c, r), rs.types[c]));
        rs.rows.push_back(std::move(row));
    }
    return rs;
}

}  // namespace

ResultSet run_duckdb(const Table& table, const LogicalQuery& q) {
    duckdb::DuckDB db(nullptr);  // in-memory
    duckdb::Connection con(db);

    load_table(con, table);

    const std::string sql = select_sql(kTable, table.schema(), q);
    auto result = con.Query(sql);
    if (result->HasError())
        throw DuckDBError(result->GetError() + "  [sql: " + sql + "]");

    // Result column order/types are the single-source-of-truth output schema
    // (projection columns, or GROUP BY keys-then-aggregates).
    std::vector<Type> types;
    const Schema out = query_output_schema(table.schema(), q);
    for (const auto& f : out.fields) types.push_back(f.second);
    return read_result(*result, std::move(types));
}

ResultSet run_join_duckdb(const Table& probe, const Table& build,
                          const JoinQuery& jq) {
    duckdb::DuckDB db(nullptr);  // in-memory
    duckdb::Connection con(db);

    load_named_table(con, "probe", probe);
    load_named_table(con, "build", build);

    const std::string sql =
        join_sql(probe.schema(), build.schema(), jq, "probe", "build");
    auto result = con.Query(sql);
    if (result->HasError())
        throw DuckDBError(result->GetError() + "  [sql: " + sql + "]");

    std::vector<Type> types;
    const Schema out = join_output_schema(probe.schema(), build.schema());
    for (const auto& f : out.fields) types.push_back(f.second);
    return read_result(*result, std::move(types));
}

ResultSet run_plan_duckdb(const qe::plan::Plan& p) {
    duckdb::DuckDB db(nullptr);  // in-memory
    duckdb::Connection con(db);

    const PlanSql ps = render_plan_sql(p);
    for (const auto& [name, tbl] : ps.tables) load_named_table(con, name, *tbl);

    auto result = con.Query(ps.sql);
    if (result->HasError())
        throw DuckDBError(result->GetError() + "  [sql: " + ps.sql + "]");

    // Result column order/types are the plan's output schema, positionally.
    std::vector<Type> types;
    for (const auto& f : p.output_schema().fields) types.push_back(f.second);
    return read_result(*result, std::move(types));
}

}  // namespace qe::oracle

#else  // !QE_WITH_DUCKDB — keep the symbols so a stray reference fails loudly.

namespace qe::oracle {
ResultSet run_duckdb(const Table&, const LogicalQuery&) {
    throw std::logic_error(
        "run_duckdb called in a build without DuckDB (QE_WITH_DUCKDB unset)");
}
ResultSet run_join_duckdb(const Table&, const Table&, const JoinQuery&) {
    throw std::logic_error(
        "run_join_duckdb called in a build without DuckDB (QE_WITH_DUCKDB unset)");
}
ResultSet run_plan_duckdb(const qe::plan::Plan&) {
    throw std::logic_error(
        "run_plan_duckdb called in a build without DuckDB (QE_WITH_DUCKDB unset)");
}
}  // namespace qe::oracle

#endif
