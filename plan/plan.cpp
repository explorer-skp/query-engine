//  WP-8: plan node construction + validation, lowering, and the printable form.
//  See plan/plan.h for the surface and the design notes (column-ref policy, Table
//  lifetime, the engine/SQL single-source-of-truth).
#include "plan/plan.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "ops/filter.h"

namespace qe::plan {
namespace {

using qe::expr::Expr;
using qe::expr::Node;
using qe::expr::NodeKind;

const char* type_name(Type t) {
    switch (t) {
        case Type::I32: return "I32";
        case Type::I64: return "I64";
        case Type::F64: return "F64";
        case Type::BOOL: return "BOOL";
        case Type::TS: return "TS";
    }
    return "?";
}

// Validate every Column reference inside `e` against `schema`: the index must be
// in range and the node's recorded type must match the schema's column type
// (expr builders record both at build time). Throws std::invalid_argument so the
// builder fails fast as the tree is assembled.
void validate_expr(const Expr& e, const Schema& schema) {
    const Node& n = e.node();
    if (n.kind == NodeKind::Column) {
        if (n.col_index >= schema.fields.size())
            throw std::invalid_argument(
                "plan: expression column index " + std::to_string(n.col_index) +
                " out of range (schema has " +
                std::to_string(schema.fields.size()) + " columns)");
        if (schema.fields[n.col_index].second != n.type)
            throw std::invalid_argument(
                "plan: expression column " + std::to_string(n.col_index) +
                " type mismatch (expr says " + type_name(n.type) +
                ", schema says " + type_name(schema.fields[n.col_index].second) +
                ")");
    }
    for (const Expr& c : n.children) validate_expr(c, schema);
}

void require_index(std::uint32_t idx, const Schema& schema, const char* what) {
    if (idx >= schema.fields.size())
        throw std::invalid_argument(
            std::string("plan: ") + what + " index " + std::to_string(idx) +
            " out of range (schema has " +
            std::to_string(schema.fields.size()) + " columns)");
}

std::shared_ptr<PlanNode> make_node(PlanKind kind) {
    auto n = std::make_shared<PlanNode>();
    n->kind = kind;
    return n;
}

// Render one expression in a compact, deterministic infix form for to_string(),
// resolving column refs to the child schema's names. Independent of the SQL
// renderer (oracle/) — this is the plan's own human-facing form.
std::string display_expr(const Expr& e, const Schema& schema) {
    const Node& n = e.node();
    switch (n.kind) {
        case NodeKind::Column:
            return n.col_index < schema.fields.size()
                       ? schema.fields[n.col_index].first
                       : ("c?" + std::to_string(n.col_index));
        case NodeKind::Literal: {
            const auto& s = n.literal;
            if (s.is_null) return std::string("NULL:") + type_name(s.type);
            std::ostringstream os;
            switch (s.type) {
                case Type::F64: os.precision(17); os << s.f; break;
                case Type::BOOL: os << (s.i ? "true" : "false"); break;
                default: os << s.i; break;
            }
            return os.str();
        }
        case NodeKind::Arith: {
            const char* op = "?";
            switch (n.arith_op) {
                case qe::expr::ArithOp::Add: op = "+"; break;
                case qe::expr::ArithOp::Sub: op = "-"; break;
                case qe::expr::ArithOp::Mul: op = "*"; break;
                case qe::expr::ArithOp::Div: op = "/"; break;
                case qe::expr::ArithOp::Mod: op = "%"; break;
            }
            return "(" + display_expr(n.children[0], schema) + " " + op + " " +
                   display_expr(n.children[1], schema) + ")";
        }
        case NodeKind::Cmp: {
            const char* op = "?";
            switch (n.cmp_op) {
                case qe::expr::CmpOp::Lt: op = "<"; break;
                case qe::expr::CmpOp::Le: op = "<="; break;
                case qe::expr::CmpOp::Gt: op = ">"; break;
                case qe::expr::CmpOp::Ge: op = ">="; break;
                case qe::expr::CmpOp::Eq: op = "=="; break;
                case qe::expr::CmpOp::Ne: op = "!="; break;
            }
            return "(" + display_expr(n.children[0], schema) + " " + op + " " +
                   display_expr(n.children[1], schema) + ")";
        }
        case NodeKind::Logic:
            switch (n.logic_op) {
                case qe::expr::LogicOp::And:
                    return "(" + display_expr(n.children[0], schema) + " AND " +
                           display_expr(n.children[1], schema) + ")";
                case qe::expr::LogicOp::Or:
                    return "(" + display_expr(n.children[0], schema) + " OR " +
                           display_expr(n.children[1], schema) + ")";
                case qe::expr::LogicOp::Not:
                    return "(NOT " + display_expr(n.children[0], schema) + ")";
            }
            return "?";
        case NodeKind::Cast:
            return "CAST(" + display_expr(n.children[0], schema) + " AS " +
                   type_name(n.cast_to) + ")";
    }
    return "?";
}

const char* agg_name(AggFunc f) {
    switch (f) {
        case AggFunc::CountStar: return "COUNT(*)";
        case AggFunc::Count: return "COUNT";
        case AggFunc::Sum: return "SUM";
        case AggFunc::Min: return "MIN";
        case AggFunc::Max: return "MAX";
        case AggFunc::Avg: return "AVG";
    }
    return "?";
}

std::string schema_str(const Schema& s) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < s.fields.size(); ++i) {
        if (i) os << ", ";
        os << s.fields[i].first << ":" << type_name(s.fields[i].second);
    }
    os << "]";
    return os.str();
}

// Recursive pretty-printer. `depth` controls 2-space indentation.
void print_node(const Plan& p, int depth, std::ostream& os) {
    const PlanNode& n = p.node();
    const std::string pad(static_cast<std::size_t>(depth) * 2, ' ');
    os << pad;
    switch (n.kind) {
        case PlanKind::Scan:
            os << "Scan " << schema_str(n.table->schema());
            break;
        case PlanKind::Filter:
            os << "Filter "
               << display_expr(n.predicate, n.children[0].output_schema());
            break;
        case PlanKind::Project: {
            const Schema& cs = n.children[0].output_schema();
            os << "Project [";
            for (std::size_t i = 0; i < n.projections.size(); ++i) {
                if (i) os << ", ";
                os << n.projections[i].name << "="
                   << display_expr(n.projections[i].expr, cs);
            }
            os << "]";
            break;
        }
        case PlanKind::Aggregate: {
            const Schema& cs = n.children[0].output_schema();
            os << "Aggregate keys=[";
            for (std::size_t i = 0; i < n.group_keys.size(); ++i) {
                if (i) os << ", ";
                os << cs.fields[n.group_keys[i]].first;
            }
            os << "] aggs=[";
            for (std::size_t i = 0; i < n.aggs.size(); ++i) {
                if (i) os << ", ";
                const AggSpec& a = n.aggs[i];
                os << agg_name(a.func);
                if (a.func != AggFunc::CountStar)
                    os << "(" << cs.fields[a.input_col].first << ")";
                os << " AS " << a.out_name;
            }
            os << "]";
            break;
        }
        case PlanKind::Join: {
            os << "Join " << (n.join_type == JoinType::Left ? "LEFT" : "INNER")
               << " left_keys=[";
            for (std::size_t i = 0; i < n.left_keys.size(); ++i)
                os << (i ? "," : "") << n.left_keys[i];
            os << "] right_keys=[";
            for (std::size_t i = 0; i < n.right_keys.size(); ++i)
                os << (i ? "," : "") << n.right_keys[i];
            os << "]";
            break;
        }
        case PlanKind::Sort: {
            os << "Sort [";
            for (std::size_t i = 0; i < n.sort_keys.size(); ++i) {
                if (i) os << ", ";
                const SortKey& k = n.sort_keys[i];
                os << k.col << (k.dir == SortDir::Desc ? " DESC" : " ASC")
                   << (k.nulls == NullOrder::First ? " NULLS FIRST"
                                                   : " NULLS LAST");
            }
            os << "]";
            break;
        }
    }
    os << "\n";
    for (const Plan& c : n.children) print_node(c, depth + 1, os);
}

}  // namespace

std::uint32_t ColRef::resolve(const Schema& schema) const {
    if (!by_name) {
        if (index >= schema.fields.size())
            throw std::invalid_argument(
                "plan: column index " + std::to_string(index) +
                " out of range (schema has " +
                std::to_string(schema.fields.size()) + " columns)");
        return index;
    }
    for (std::uint32_t i = 0; i < schema.fields.size(); ++i)
        if (schema.fields[i].first == name)
            return i;
    throw std::invalid_argument("plan: unknown column name '" + name + "'");
}

// ---- node builders (each validates + computes the output schema) ------------

PlanBuilder scan(const Table& table) {
    auto n = make_node(PlanKind::Scan);
    n->table = &table;
    n->out_schema = table.schema();
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::filter(Expr predicate) const {
    const Schema& cs = plan_.output_schema();
    if (!predicate)
        throw std::invalid_argument("plan: filter predicate is empty");
    if (predicate.type() != Type::BOOL)
        throw std::invalid_argument(
            std::string("plan: filter predicate must be BOOL, got ") +
            type_name(predicate.type()));
    validate_expr(predicate, cs);
    auto n = make_node(PlanKind::Filter);
    n->children.push_back(plan_);
    n->predicate = std::move(predicate);
    n->out_schema = cs;  // filter never changes the schema
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::project(std::vector<Projection> projections) const {
    const Schema& cs = plan_.output_schema();
    if (projections.empty())
        throw std::invalid_argument("plan: project needs >=1 column");
    Schema out;
    out.fields.reserve(projections.size());
    for (const Projection& p : projections) {
        validate_expr(p.expr, cs);
        out.fields.emplace_back(p.name, p.expr.type());
    }
    auto n = make_node(PlanKind::Project);
    n->children.push_back(plan_);
    n->projections = std::move(projections);
    n->out_schema = std::move(out);
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::aggregate(std::vector<ColRef> keys,
                                   std::vector<AggSpec> aggs) const {
    const Schema& cs = plan_.output_schema();
    if (aggs.empty())
        throw std::invalid_argument("plan: aggregate needs >=1 aggregate");
    std::vector<std::uint32_t> key_idx;
    key_idx.reserve(keys.size());
    Schema out;
    for (const ColRef& k : keys) {
        const std::uint32_t i = k.resolve(cs);
        key_idx.push_back(i);
        out.fields.emplace_back(cs.fields[i].first, cs.fields[i].second);
    }
    for (const AggSpec& a : aggs) {
        Type in = Type::I64;
        if (a.func != AggFunc::CountStar) {
            require_index(a.input_col, cs, "aggregate input column");
            in = cs.fields[a.input_col].second;
        }
        out.fields.emplace_back(a.out_name, agg_result_type(a.func, in));
    }
    auto n = make_node(PlanKind::Aggregate);
    n->children.push_back(plan_);
    n->group_keys = std::move(key_idx);
    n->aggs = std::move(aggs);
    n->out_schema = std::move(out);
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::join(const Plan& build, std::vector<ColRef> left_keys,
                              std::vector<ColRef> right_keys,
                              JoinType type) const {
    const Schema& ls = plan_.output_schema();
    const Schema& rs = build.output_schema();
    if (left_keys.size() != right_keys.size() || left_keys.empty())
        throw std::invalid_argument(
            "plan: join needs an equal, non-empty number of left/right keys");
    std::vector<std::uint32_t> lk, rk;
    lk.reserve(left_keys.size());
    rk.reserve(right_keys.size());
    for (std::size_t i = 0; i < left_keys.size(); ++i) {
        const std::uint32_t li = left_keys[i].resolve(ls);
        const std::uint32_t ri = right_keys[i].resolve(rs);
        if (ls.fields[li].second != rs.fields[ri].second)
            throw std::invalid_argument(
                "plan: join key " + std::to_string(i) + " type mismatch (left " +
                type_name(ls.fields[li].second) + " vs right " +
                type_name(rs.fields[ri].second) + ")");
        lk.push_back(li);
        rk.push_back(ri);
    }
    // Output = all probe (left) columns then all build (right) columns.
    Schema out;
    out.fields.reserve(ls.fields.size() + rs.fields.size());
    for (const auto& f : ls.fields) out.fields.push_back(f);
    for (const auto& f : rs.fields) out.fields.push_back(f);
    auto n = make_node(PlanKind::Join);
    n->children.push_back(plan_);  // [0] = probe / left
    n->children.push_back(build);  // [1] = build / right
    n->left_keys = std::move(lk);
    n->right_keys = std::move(rk);
    n->join_type = type;
    n->out_schema = std::move(out);
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::sort(std::vector<SortKey> keys) const {
    const Schema& cs = plan_.output_schema();
    if (keys.empty()) throw std::invalid_argument("plan: sort needs >=1 key");
    for (const SortKey& k : keys) require_index(k.col, cs, "sort key");
    auto n = make_node(PlanKind::Sort);
    n->children.push_back(plan_);
    n->sort_keys = std::move(keys);
    n->out_schema = cs;  // sort reorders rows, never columns/types
    return PlanBuilder(Plan(std::move(n)));
}

PlanBuilder PlanBuilder::sort(std::vector<SortBy> keys) const {
    const Schema& cs = plan_.output_schema();
    std::vector<SortKey> resolved;
    resolved.reserve(keys.size());
    for (const SortBy& k : keys)
        resolved.push_back(SortKey{k.col.resolve(cs), k.dir, k.nulls});
    return sort(std::move(resolved));
}

// ---- lowering ---------------------------------------------------------------

std::unique_ptr<Operator> Plan::lower(std::size_t batch_size) const {
    const PlanNode& n = *node_;
    switch (n.kind) {
        case PlanKind::Scan:
            return std::make_unique<Scan>(*n.table, batch_size);
        case PlanKind::Filter:
            return std::make_unique<Filter>(n.children[0].lower(batch_size),
                                            n.predicate);
        case PlanKind::Project:
            return std::make_unique<Project>(n.children[0].lower(batch_size),
                                             n.projections);
        case PlanKind::Aggregate:
            return std::make_unique<Aggregate>(n.children[0].lower(batch_size),
                                               n.group_keys, n.aggs);
        case PlanKind::Join:
            // children[0] = probe/left, children[1] = build/right — the exact
            // order oracle/logical_query.cpp::build_join_pipeline uses.
            return std::make_unique<HashJoin>(
                n.children[0].lower(batch_size),
                n.children[1].lower(batch_size), n.left_keys, n.right_keys,
                n.join_type);
        case PlanKind::Sort:
            return std::make_unique<Sort>(n.children[0].lower(batch_size),
                                          n.sort_keys);
    }
    throw std::logic_error("plan: unhandled PlanKind in lower()");
}

std::string Plan::to_string() const {
    std::ostringstream os;
    print_node(*this, 0, os);
    return os.str();
}

}  // namespace qe::plan
