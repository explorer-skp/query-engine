//  WP-8: plan::Plan -> SQL. See oracle/plan_sql.h. Reuses the shared helpers in
//  oracle/sql_render.h (expr_to_sql, agg_call_sql, order_by_sql, sql_type); the
//  composition logic (subquery tree, o-named columns) is new because the existing
//  flat select_sql/join_sql assume a single base table, not nested subqueries.
#include "oracle/plan_sql.h"

#include <map>
#include <sstream>
#include <string>

#include "oracle/sql_render.h"

namespace qe::oracle {
namespace {

using qe::plan::Plan;
using qe::plan::PlanKind;
using qe::plan::PlanNode;

// Build a schema whose names are the canonical positional o0..oN (types from `s`).
// Passed to expr_to_sql / agg_call_sql so an index i resolves to the name "oi" of
// the rendered child subquery.
Schema o_schema(const Schema& s) {
    Schema o;
    o.fields.reserve(s.fields.size());
    for (std::size_t i = 0; i < s.fields.size(); ++i)
        o.fields.emplace_back("o" + std::to_string(i), s.fields[i].second);
    return o;
}

struct Ctx {
    std::vector<std::pair<std::string, const qe::Table*>> tables;
    std::map<const qe::Table*, std::string> names;
    int alias = 0;

    // SQL base-table name for `t` (deduped: the same Table* loads once; a self-
    // join's two scans share the base and are disambiguated by subquery aliases).
    const std::string& base_name(const qe::Table* t) {
        auto it = names.find(t);
        if (it != names.end()) return it->second;
        std::string nm = "base" + std::to_string(names.size());
        auto [ins, _] = names.emplace(t, std::move(nm));
        tables.emplace_back(ins->second, t);
        return ins->second;
    }
    std::string next_alias() { return "r" + std::to_string(alias++); }
};

std::string render(const Plan& p, Ctx& ctx);

// "SELECT o0 AS o0, ..., oK AS oK" — the identity projection used by pass-through
// nodes (Filter/Sort) so their output is also canonically o-named.
std::string passthrough_cols(std::size_t ncol) {
    std::ostringstream os;
    for (std::size_t i = 0; i < ncol; ++i) {
        if (i) os << ", ";
        os << "o" << i << " AS o" << i;
    }
    return os.str();
}

std::string render_scan(const PlanNode& n, Ctx& ctx) {
    const Schema& s = n.table->schema();
    const std::string& base = ctx.base_name(n.table);
    std::ostringstream os;
    os << "SELECT ";
    for (std::size_t i = 0; i < s.fields.size(); ++i) {
        if (i) os << ", ";
        os << s.fields[i].first << " AS o" << i;
    }
    os << " FROM " << base;
    return os.str();
}

std::string render_filter(const PlanNode& n, Ctx& ctx) {
    const Plan& child = n.children[0];
    const std::string csql = render(child, ctx);
    const Schema cos = o_schema(child.output_schema());
    std::ostringstream os;
    os << "SELECT " << passthrough_cols(cos.fields.size()) << " FROM (" << csql
       << ") AS " << ctx.next_alias() << " WHERE "
       << expr_to_sql(n.predicate, cos);
    return os.str();
}

std::string render_project(const PlanNode& n, Ctx& ctx) {
    const Plan& child = n.children[0];
    const std::string csql = render(child, ctx);
    const Schema cos = o_schema(child.output_schema());
    std::ostringstream os;
    os << "SELECT ";
    for (std::size_t i = 0; i < n.projections.size(); ++i) {
        if (i) os << ", ";
        const auto& pr = n.projections[i];
        os << "CAST((" << expr_to_sql(pr.expr, cos) << ") AS "
           << sql_type(pr.expr.type()) << ") AS o" << i;
    }
    os << " FROM (" << csql << ") AS " << ctx.next_alias();
    return os.str();
}

std::string render_aggregate(const PlanNode& n, Ctx& ctx) {
    const Plan& child = n.children[0];
    const std::string csql = render(child, ctx);
    const Schema cos = o_schema(child.output_schema());
    std::ostringstream os;
    os << "SELECT ";
    std::size_t o = 0;
    for (std::uint32_t kc : n.group_keys) {
        if (o) os << ", ";
        os << "CAST((o" << kc << ") AS " << sql_type(cos.fields[kc].second)
           << ") AS o" << o;
        ++o;
    }
    for (const AggSpec& a : n.aggs) {
        if (o) os << ", ";
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : cos.fields[a.input_col].second;
        os << "CAST((" << agg_call_sql(a, cos) << ") AS "
           << sql_type(agg_result_type(a.func, in)) << ") AS o" << o;
        ++o;
    }
    os << " FROM (" << csql << ") AS " << ctx.next_alias();
    if (!n.group_keys.empty()) {
        os << " GROUP BY ";
        for (std::size_t i = 0; i < n.group_keys.size(); ++i) {
            if (i) os << ", ";
            os << "o" << n.group_keys[i];
        }
    }
    return os.str();
}

std::string render_join(const PlanNode& n, Ctx& ctx) {
    const Plan& left = n.children[0];   // probe
    const Plan& right = n.children[1];  // build
    const std::string lsql = render(left, ctx);
    const std::string rsql = render(right, ctx);
    const Schema& ls = left.output_schema();
    const Schema& rs = right.output_schema();
    const std::string pa = ctx.next_alias();  // probe alias
    const std::string ba = ctx.next_alias();  // build alias
    std::ostringstream os;
    os << "SELECT ";
    std::size_t o = 0;
    for (std::size_t i = 0; i < ls.fields.size(); ++i) {
        if (o) os << ", ";
        os << "CAST(" << pa << ".o" << i << " AS " << sql_type(ls.fields[i].second)
           << ") AS o" << o;
        ++o;
    }
    for (std::size_t j = 0; j < rs.fields.size(); ++j) {
        if (o) os << ", ";
        os << "CAST(" << ba << ".o" << j << " AS " << sql_type(rs.fields[j].second)
           << ") AS o" << o;
        ++o;
    }
    os << " FROM (" << lsql << ") AS " << pa << " "
       << (n.join_type == JoinType::Left ? "LEFT JOIN " : "JOIN ") << "(" << rsql
       << ") AS " << ba << " ON ";
    for (std::size_t i = 0; i < n.left_keys.size(); ++i) {
        if (i) os << " AND ";
        os << pa << ".o" << n.left_keys[i] << " = " << ba << ".o"
           << n.right_keys[i];
    }
    return os.str();
}

std::string render_sort(const PlanNode& n, Ctx& ctx) {
    const Plan& child = n.children[0];
    const std::string csql = render(child, ctx);
    const std::size_t ncol = child.output_schema().fields.size();
    std::ostringstream os;
    os << "SELECT " << passthrough_cols(ncol) << " FROM (" << csql << ") AS "
       << ctx.next_alias() << " " << order_by_sql(n.sort_keys);
    return os.str();
}

std::string render(const Plan& p, Ctx& ctx) {
    const PlanNode& n = p.node();
    switch (n.kind) {
        case PlanKind::Scan: return render_scan(n, ctx);
        case PlanKind::Filter: return render_filter(n, ctx);
        case PlanKind::Project: return render_project(n, ctx);
        case PlanKind::Aggregate: return render_aggregate(n, ctx);
        case PlanKind::Join: return render_join(n, ctx);
        case PlanKind::Sort: return render_sort(n, ctx);
    }
    return "";  // unreachable
}

}  // namespace

PlanSql render_plan_sql(const qe::plan::Plan& p) {
    Ctx ctx;
    PlanSql out;
    out.sql = render(p, ctx);
    out.tables = std::move(ctx.tables);
    return out;
}

}  // namespace qe::oracle
