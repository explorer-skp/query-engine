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

std::string select_sql(const std::string& name, const Schema& schema,
                       const LogicalQuery& q) {
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
    return os.str();
}

}  // namespace qe::oracle
