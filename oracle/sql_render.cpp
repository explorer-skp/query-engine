//  WP-3 (seed of WP-9): LogicalQuery -> SQL rendering. See oracle/sql_render.h.
#include "oracle/sql_render.h"

#include <sstream>
#include <stdexcept>

#include "expr/expr.h"

namespace qe::oracle {

using namespace qe::expr;

std::string sql_type(Type t) {
    switch (t) {
        case Type::I32: return "INTEGER";
        case Type::I64: return "BIGINT";
        case Type::F64: return "DOUBLE";
        case Type::BOOL: return "BOOLEAN";
        case Type::TS: return "BIGINT";
        case Type::STR: return "VARCHAR";  // WP-7b: dictionary-encoded text
    }
    return "BIGINT";  // unreachable
}

namespace {

std::string arith_sql(ArithOp op, Type result_type, const std::string& a,
                      const std::string& b) {
    const char* sym = "+";
    switch (op) {
        case ArithOp::Add: sym = "+"; break;
        case ArithOp::Sub: sym = "-"; break;
        case ArithOp::Mul: sym = "*"; break;
        case ArithOp::Div:
            // Engine integer division truncates toward zero; DuckDB '/' is float
            // division, '//' is integer division. Match by operand-result type.
            sym = (result_type == Type::F64) ? "/" : "//";
            break;
        case ArithOp::Mod: sym = "%"; break;
    }
    return "(" + a + " " + sym + " " + b + ")";
}

const char* cmp_sql(CmpOp op) {
    switch (op) {
        case CmpOp::Lt: return "<";
        case CmpOp::Le: return "<=";
        case CmpOp::Gt: return ">";
        case CmpOp::Ge: return ">=";
        case CmpOp::Eq: return "=";
        case CmpOp::Ne: return "<>";
    }
    return "=";
}

std::string scalar_sql(const Scalar& s) {
    if (s.is_null) return "CAST(NULL AS " + sql_type(s.type) + ")";
    std::ostringstream os;
    switch (s.type) {
        case Type::I32:
        case Type::I64:
        case Type::TS:
            os << s.i;
            break;
        case Type::BOOL:
            os << (s.i ? "TRUE" : "FALSE");
            break;
        case Type::F64:
            // Enough precision to round-trip a double exactly.
            os.precision(17);
            os << s.f;
            break;
        case Type::STR:
            // WP-7b: the frozen Scalar carries no string payload, so STR literals
            // are never built (STR comparisons are column-vs-column). Unreachable.
            throw std::logic_error("scalar_sql: STR literals are unsupported");
    }
    return os.str();
}

}  // namespace

std::string expr_to_sql(const Expr& e, const Schema& schema) {
    const Node& n = e.node();
    switch (n.kind) {
        case NodeKind::Column: {
            if (n.col_index >= schema.fields.size())
                throw std::out_of_range("expr_to_sql: column index out of range");
            return schema.fields[n.col_index].first;
        }
        case NodeKind::Literal:
            return scalar_sql(n.literal);
        case NodeKind::Arith:
            return arith_sql(n.arith_op, n.type,
                             expr_to_sql(n.children[0], schema),
                             expr_to_sql(n.children[1], schema));
        case NodeKind::Cmp:
            return "(" + expr_to_sql(n.children[0], schema) + " " +
                   cmp_sql(n.cmp_op) + " " +
                   expr_to_sql(n.children[1], schema) + ")";
        case NodeKind::Logic:
            switch (n.logic_op) {
                case LogicOp::And:
                    return "(" + expr_to_sql(n.children[0], schema) + " AND " +
                           expr_to_sql(n.children[1], schema) + ")";
                case LogicOp::Or:
                    return "(" + expr_to_sql(n.children[0], schema) + " OR " +
                           expr_to_sql(n.children[1], schema) + ")";
                case LogicOp::Not:
                    return "(NOT " + expr_to_sql(n.children[0], schema) + ")";
            }
            break;
        case NodeKind::Cast:
            return "CAST(" + expr_to_sql(n.children[0], schema) + " AS " +
                   sql_type(n.cast_to) + ")";
    }
    throw std::logic_error("expr_to_sql: unhandled node kind");
}

std::string create_table_sql(const std::string& name, const Schema& schema) {
    std::ostringstream os;
    os << "CREATE TABLE " << name << "(";
    for (std::size_t i = 0; i < schema.fields.size(); ++i) {
        if (i) os << ", ";
        os << schema.fields[i].first << " " << sql_type(schema.fields[i].second);
    }
    os << ")";
    return os.str();
}

// The aggregate function call SQL, e.g. "SUM(c2)". CountStar is "COUNT(*)".
// Public (declared in sql_render.h) so the plan->SQL renderer reuses it.
std::string agg_call_sql(const AggSpec& a, const Schema& schema) {
    if (a.func == AggFunc::CountStar) return "COUNT(*)";
    const std::string& col = schema.fields[a.input_col].first;
    switch (a.func) {
        case AggFunc::Count: return "COUNT(" + col + ")";
        case AggFunc::Sum:   return "SUM(" + col + ")";
        case AggFunc::Min:   return "MIN(" + col + ")";
        case AggFunc::Max:   return "MAX(" + col + ")";
        case AggFunc::Avg:   return "AVG(" + col + ")";
        case AggFunc::CountStar: break;  // handled above
    }
    return "COUNT(*)";  // unreachable
}

// "ORDER BY ..." by 1-based output ordinal, explicit dir + null order. Empty when
// there are no keys.
std::string order_by_sql(const std::vector<SortKey>& keys) {
    if (keys.empty()) return "";
    std::ostringstream os;
    os << "ORDER BY ";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i) os << ", ";
        os << (keys[i].col + 1)
           << (keys[i].dir == SortDir::Desc ? " DESC" : " ASC")
           << (keys[i].nulls == NullOrder::First ? " NULLS FIRST"
                                                 : " NULLS LAST");
    }
    return os.str();
}

namespace {

// Render a group-by query. Each output column is wrapped in a CAST to the
// engine's result type so the DuckDB column type matches EXACTLY. The CAST on
// SUM is the overflow-honesty step: DuckDB's SUM(INTEGER)/SUM(BIGINT) return
// HUGEINT; casting to BIGINT makes the read-back I64 unambiguous, and the oracle
// generators bound the data so every per-group sum provably fits I64 (so the cast
// never raises). See oracle/generators.* and the WP-5 report.
std::string group_by_sql(const std::string& name, const Schema& schema,
                         const LogicalQuery& q) {
    const GroupBy& gb = *q.group_by;
    std::ostringstream os;
    os << "SELECT ";
    bool first = true;
    for (std::uint32_t kc : gb.keys) {
        if (!first) os << ", ";
        first = false;
        const auto& f = schema.fields[kc];
        os << "CAST((" << f.first << ") AS " << sql_type(f.second) << ") AS "
           << f.first;
    }
    for (const auto& a : gb.aggs) {
        if (!first) os << ", ";
        first = false;
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : schema.fields[a.input_col].second;
        const Type rt = agg_result_type(a.func, in);
        os << "CAST((" << agg_call_sql(a, schema) << ") AS " << sql_type(rt)
           << ") AS " << a.out_name;
    }
    os << " FROM " << name;
    if (q.has_filter()) os << " WHERE " << expr_to_sql(q.filter, schema);
    if (!gb.keys.empty()) {
        os << " GROUP BY ";
        for (std::size_t i = 0; i < gb.keys.size(); ++i) {
            if (i) os << ", ";
            os << schema.fields[gb.keys[i]].first;
        }
    }
    return os.str();
}

// WP-7: render an explicit ORDER BY over the query's OUTPUT columns. Keys are
// rendered by 1-based ORDINAL (output position) so they bind unambiguously to the
// SELECT list regardless of column names — the output position matches
// query_output_schema order, which is the engine's output order. NULLS FIRST/LAST
// is emitted explicitly so the result never depends on DuckDB's default null
// order. Empty when the query has no ORDER BY.
std::string order_by_clause(const LogicalQuery& q) {
    if (!q.has_order_by()) return "";
    return " " + order_by_sql(*q.order_by);  // shared renderer (WP-8)
}

}  // namespace

std::string join_sql(const Schema& probe, const Schema& build,
                     const JoinQuery& jq, const std::string& probe_name,
                     const std::string& build_name) {
    std::ostringstream os;
    os << "SELECT ";
    std::size_t o = 0;
    // Probe columns, aliased o0..  by position; CAST to the engine's own type.
    for (const auto& f : probe.fields) {
        if (o) os << ", ";
        os << "CAST(p." << f.first << " AS " << sql_type(f.second) << ") AS o"
           << o;
        ++o;
    }
    // Build columns next.
    for (const auto& f : build.fields) {
        if (o) os << ", ";
        os << "CAST(b." << f.first << " AS " << sql_type(f.second) << ") AS o"
           << o;
        ++o;
    }
    os << " FROM " << probe_name << " AS p "
       << (jq.type == JoinType::Left ? "LEFT JOIN " : "JOIN ") << build_name
       << " AS b ON ";
    for (std::size_t i = 0; i < jq.probe_keys.size(); ++i) {
        if (i) os << " AND ";
        os << "p." << probe.fields[jq.probe_keys[i]].first << " = b."
           << build.fields[jq.build_keys[i]].first;
    }
    return os.str();
}

std::string select_sql(const std::string& name, const Schema& schema,
                       const LogicalQuery& q) {
    if (q.has_group_by())
        return group_by_sql(name, schema, q) + order_by_clause(q);
    std::ostringstream os;
    os << "SELECT ";
    for (std::size_t i = 0; i < q.projections.size(); ++i) {
        if (i) os << ", ";
        const auto& p = q.projections[i];
        // Wrap in a CAST to the expr's own type so the DuckDB result column type
        // matches the engine's exactly (identity/widening cast; never raises).
        os << "CAST((" << expr_to_sql(p.expr, schema) << ") AS "
           << sql_type(p.expr.type()) << ") AS " << p.name;
    }
    os << " FROM " << name;
    if (q.has_filter()) os << " WHERE " << expr_to_sql(q.filter, schema);
    os << order_by_clause(q);  // WP-7: explicit ORDER BY when present
    return os.str();
}

}  // namespace qe::oracle
