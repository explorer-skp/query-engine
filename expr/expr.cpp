//  WP-2: expression builders + type inference/validation (see expr/expr.h).
//
//  Every builder validates operand types and computes the result type at build
//  time, throwing std::invalid_argument on a type error. Numeric binary ops
//  AUTO-INSERT promotion casts so the evaluator and the op×type kernels only
//  ever see two operands of one identical physical type — the matrix stays
//  monomorphic and no kernel does mixed-type arithmetic.

#include "expr/expr.h"

#include <stdexcept>
#include <string>

namespace qe::expr {
namespace {

bool is_numeric(Type t) {
    return t == Type::I32 || t == Type::I64 || t == Type::F64;
}

// Numeric promotion rank: F64 > I64 > I32. The promoted type of two numerics is
// the higher-ranked one (matches the usual widening; TS/BOOL are not numeric and
// must be cast explicitly before arithmetic).
int numeric_rank(Type t) {
    switch (t) {
        case Type::I32:
            return 1;
        case Type::I64:
            return 2;
        case Type::F64:
            return 3;
        default:
            return 0;
    }
}

Type promote_numeric(Type a, Type b) {
    return numeric_rank(a) >= numeric_rank(b) ? a : b;
}

std::shared_ptr<Node> make_node(NodeKind k, Type t) {
    auto n = std::make_shared<Node>();
    n->kind = k;
    n->type = t;
    return n;
}

const char* type_name(Type t) {
    switch (t) {
        case Type::I32:
            return "I32";
        case Type::I64:
            return "I64";
        case Type::F64:
            return "F64";
        case Type::BOOL:
            return "BOOL";
        case Type::TS:
            return "TS";
    }
    return "?";
}

}  // namespace

Expr col(Type type, std::uint32_t index) {
    auto n = make_node(NodeKind::Column, type);
    n->col_index = index;
    return Expr(std::move(n));
}

Expr col(const Schema& schema, std::string_view name) {
    for (std::size_t i = 0; i < schema.fields.size(); ++i) {
        if (schema.fields[i].first == name) {
            return col(schema.fields[i].second, static_cast<std::uint32_t>(i));
        }
    }
    throw std::invalid_argument("col: no column named '" + std::string(name) +
                                "' in schema");
}

Expr lit(Scalar s) {
    auto n = make_node(NodeKind::Literal, s.type);
    n->literal = s;
    return Expr(std::move(n));
}

Expr cast(Expr e, Type to) {
    // Identity cast collapses to the child (no node needed).
    if (e.type() == to) return e;
    // Every cross-type pair among the five Types is meaningful in this engine
    // (numeric<->numeric, ->/<-BOOL via !=0/{0,1}, TS<->I64 reinterpret, etc.).
    auto n = make_node(NodeKind::Cast, to);
    n->cast_to = to;
    n->children.push_back(std::move(e));
    return Expr(std::move(n));
}

Expr arith(ArithOp op, Expr a, Expr b) {
    if (!is_numeric(a.type()) || !is_numeric(b.type())) {
        throw std::invalid_argument(
            std::string("arith: operands must be numeric (I32/I64/F64); got ") +
            type_name(a.type()) + " and " + type_name(b.type()) +
            " (cast TS/BOOL explicitly)");
    }
    const Type common = promote_numeric(a.type(), b.type());
    auto n = make_node(NodeKind::Arith, common);
    n->arith_op = op;
    n->children.push_back(cast(std::move(a), common));
    n->children.push_back(cast(std::move(b), common));
    return Expr(std::move(n));
}

Expr add(Expr a, Expr b) { return arith(ArithOp::Add, std::move(a), std::move(b)); }
Expr sub(Expr a, Expr b) { return arith(ArithOp::Sub, std::move(a), std::move(b)); }
Expr mul(Expr a, Expr b) { return arith(ArithOp::Mul, std::move(a), std::move(b)); }
Expr div(Expr a, Expr b) { return arith(ArithOp::Div, std::move(a), std::move(b)); }
Expr mod(Expr a, Expr b) { return arith(ArithOp::Mod, std::move(a), std::move(b)); }

Expr cmp(CmpOp op, Expr a, Expr b) {
    Type common;
    if (is_numeric(a.type()) && is_numeric(b.type())) {
        common = promote_numeric(a.type(), b.type());
    } else if (a.type() == b.type()) {
        // BOOL vs BOOL or TS vs TS — compare in place.
        common = a.type();
    } else {
        throw std::invalid_argument(
            std::string("cmp: incomparable operand types ") +
            type_name(a.type()) + " and " + type_name(b.type()));
    }
    auto n = make_node(NodeKind::Cmp, Type::BOOL);
    n->cmp_op = op;
    n->children.push_back(cast(std::move(a), common));
    n->children.push_back(cast(std::move(b), common));
    return Expr(std::move(n));
}

Expr lt(Expr a, Expr b) { return cmp(CmpOp::Lt, std::move(a), std::move(b)); }
Expr le(Expr a, Expr b) { return cmp(CmpOp::Le, std::move(a), std::move(b)); }
Expr gt(Expr a, Expr b) { return cmp(CmpOp::Gt, std::move(a), std::move(b)); }
Expr ge(Expr a, Expr b) { return cmp(CmpOp::Ge, std::move(a), std::move(b)); }
Expr eq(Expr a, Expr b) { return cmp(CmpOp::Eq, std::move(a), std::move(b)); }
Expr ne(Expr a, Expr b) { return cmp(CmpOp::Ne, std::move(a), std::move(b)); }

Expr logic_and(Expr a, Expr b) {
    if (a.type() != Type::BOOL || b.type() != Type::BOOL) {
        throw std::invalid_argument("logic_and: operands must be BOOL");
    }
    auto n = make_node(NodeKind::Logic, Type::BOOL);
    n->logic_op = LogicOp::And;
    n->children.push_back(std::move(a));
    n->children.push_back(std::move(b));
    return Expr(std::move(n));
}

Expr logic_or(Expr a, Expr b) {
    if (a.type() != Type::BOOL || b.type() != Type::BOOL) {
        throw std::invalid_argument("logic_or: operands must be BOOL");
    }
    auto n = make_node(NodeKind::Logic, Type::BOOL);
    n->logic_op = LogicOp::Or;
    n->children.push_back(std::move(a));
    n->children.push_back(std::move(b));
    return Expr(std::move(n));
}

Expr logic_not(Expr a) {
    if (a.type() != Type::BOOL) {
        throw std::invalid_argument("logic_not: operand must be BOOL");
    }
    auto n = make_node(NodeKind::Logic, Type::BOOL);
    n->logic_op = LogicOp::Not;
    n->children.push_back(std::move(a));
    return Expr(std::move(n));
}

}  // namespace qe::expr
