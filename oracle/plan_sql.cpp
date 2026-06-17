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

// WP-12 (additive): render an AsofJoin to DuckDB `ASOF [LEFT] JOIN`. The match is
// the greatest build timestamp <= probe timestamp (the inequality `pa.t >= ba.t`),
// among rows agreeing on the equality keys. Output columns are probe-then-build,
// o-aliased by POSITION (same scheme as render_join), each CAST to its engine type.
//
// TOLERANCE (verified vs DuckDB v1.1.3): "nearest preceding, then drop if t - tb >
// tol" (= pandas merge_asof window; nearest-then-check and nearest-within-window
// coincide for a backward as-of). For INNER this is an EXTRA ON conjunct
// `(pa.t - ba.t) <= tol` (which correctly drops out-of-window rows). For LEFT the
// extra ON conjunct would WRONGLY drop the probe row, so we keep the plain ASOF
// LEFT JOIN and NULL-MASK each build column with
// `CASE WHEN (pa.t - ba.t) <= tol THEN ba.oj END` (probe row stays, build cols NULL
// when out of window) — matching the engine's LEFT semantics exactly.
std::string render_asof(const PlanNode& n, Ctx& ctx) {
    const Plan& left = n.children[0];   // probe
    const Plan& right = n.children[1];  // build
    const std::string lsql = render(left, ctx);
    const std::string rsql = render(right, ctx);
    const Schema& ls = left.output_schema();
    const Schema& rs = right.output_schema();
    const std::string pa = ctx.next_alias();  // probe alias
    const std::string ba = ctx.next_alias();  // build alias
    const bool is_left = n.asof_type == tsx::AsofType::Left;

    // The "(pa.tcol - ba.tcol) <= tol" window predicate (only when tolerance set).
    std::ostringstream win;
    if (n.asof_tolerance)
        win << "(" << pa << ".o" << n.asof_left_time << " - " << ba << ".o"
            << n.asof_right_time << ") <= " << *n.asof_tolerance;
    const std::string window = win.str();
    const bool mask_build = n.asof_tolerance && is_left;  // LEFT => CASE-mask

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
        os << "CAST(";
        if (mask_build)
            os << "CASE WHEN " << window << " THEN " << ba << ".o" << j << " END";
        else
            os << ba << ".o" << j;
        os << " AS " << sql_type(rs.fields[j].second) << ") AS o" << o;
        ++o;
    }
    os << " FROM (" << lsql << ") AS " << pa << " "
       << (is_left ? "ASOF LEFT JOIN " : "ASOF JOIN ") << "(" << rsql << ") AS "
       << ba << " ON ";
    // ON conjuncts: equality keys, then the single backward inequality, then (for
    // INNER + tolerance) the window as an extra conjunct.
    bool first = true;
    for (std::size_t i = 0; i < n.asof_left_keys.size(); ++i) {
        os << (first ? "" : " AND ") << pa << ".o" << n.asof_left_keys[i] << " = "
           << ba << ".o" << n.asof_right_keys[i];
        first = false;
    }
    os << (first ? "" : " AND ") << pa << ".o" << n.asof_left_time
       << " >= " << ba << ".o" << n.asof_right_time;
    if (n.asof_tolerance && !is_left) os << " AND " << window;
    return os.str();
}

// WP-13 (additive): render a Window node to DuckDB.
//  * TUMBLING => a GROUP BY over the partition keys and the integer time bucket. The
//    bucket lower edge is `(o_t - (o_t % W))` — integer `t - (t % W)` is EXACT
//    bucketing for t >= 0, W > 0 (the generators keep timestamps >= 0), and avoids
//    FLOOR/double drift. Output columns are the keys, then the bucket (CAST to the
//    timestamp type), then each aggregate (CAST to its agg_result_type) — o-aliased
//    by POSITION, the same scheme as render_aggregate.
//  * SLIDING => one row per input row: the child columns pass through, then each
//    aggregate as a WINDOW function `<agg> OVER (PARTITION BY <keys> ORDER BY o_t
//    ROWS BETWEEN P PRECEDING AND CURRENT ROW)`, CAST to its agg_result_type.
//    COUNT(*) renders as `COUNT(*) OVER (…)`. With zero partition keys the PARTITION
//    BY clause is omitted (a single global window).
std::string render_window(const PlanNode& n, Ctx& ctx) {
    const Plan& child = n.children[0];
    const std::string csql = render(child, ctx);
    const Schema cos = o_schema(child.output_schema());
    const std::string alias = ctx.next_alias();

    // The bucket / order column is the child's o<window_time> column.
    const std::string ot = "o" + std::to_string(n.window_time);
    const Type ttype = cos.fields[n.window_time].second;

    std::ostringstream os;
    os << "SELECT ";
    if (n.window_mode == tsx::WindowMode::Tumbling) {
        std::ostringstream bucket;
        bucket << "(" << ot << " - (" << ot << " % " << n.window_param << "))";
        std::size_t o = 0;
        for (std::uint32_t kc : n.window_keys) {
            if (o) os << ", ";
            os << "CAST((o" << kc << ") AS " << sql_type(cos.fields[kc].second)
               << ") AS o" << o;
            ++o;
        }
        if (o) os << ", ";
        os << "CAST(" << bucket.str() << " AS " << sql_type(ttype) << ") AS o" << o;
        ++o;
        for (const AggSpec& a : n.window_aggs) {
            os << ", ";
            const Type in = (a.func == AggFunc::CountStar)
                                ? Type::I64
                                : cos.fields[a.input_col].second;
            os << "CAST((" << agg_call_sql(a, cos) << ") AS "
               << sql_type(agg_result_type(a.func, in)) << ") AS o" << o;
            ++o;
        }
        os << " FROM (" << csql << ") AS " << alias << " GROUP BY ";
        for (std::uint32_t kc : n.window_keys) os << "o" << kc << ", ";
        os << bucket.str();
        return os.str();
    }

    // SLIDING.
    std::ostringstream over;
    over << " OVER (";
    if (!n.window_keys.empty()) {
        over << "PARTITION BY ";
        for (std::size_t i = 0; i < n.window_keys.size(); ++i)
            over << (i ? ", " : "") << "o" << n.window_keys[i];
        over << " ";
    }
    over << "ORDER BY " << ot << " ROWS BETWEEN " << n.window_param
         << " PRECEDING AND CURRENT ROW)";
    const std::string over_clause = over.str();

    std::size_t o = 0;
    for (std::size_t i = 0; i < cos.fields.size(); ++i) {
        if (o) os << ", ";
        os << "o" << i << " AS o" << o;
        ++o;
    }
    for (const AggSpec& a : n.window_aggs) {
        os << ", ";
        const Type in = (a.func == AggFunc::CountStar)
                            ? Type::I64
                            : cos.fields[a.input_col].second;
        os << "CAST((" << agg_call_sql(a, cos) << over_clause << ") AS "
           << sql_type(agg_result_type(a.func, in)) << ") AS o" << o;
        ++o;
    }
    os << " FROM (" << csql << ") AS " << alias;
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

// WP-14 (additive): render a CompressedScan to the SAME SQL as a plain scan of its
// INDEPENDENT reference Table (n.ctable_ref) — i.e. it loads the DECOMPRESSED source
// into DuckDB. The engine path decodes n.ctable's compressed bytes independently, so
// agreement between the two requires the codec to be truly lossless (the non-circular
// split: the oracle never touches n.ctable; lowering never touches n.ctable_ref).
std::string render_compressed_scan(const PlanNode& n, Ctx& ctx) {
    const Schema& s = n.ctable_ref->schema();
    const std::string& base = ctx.base_name(n.ctable_ref);
    std::ostringstream os;
    os << "SELECT ";
    for (std::size_t i = 0; i < s.fields.size(); ++i) {
        if (i) os << ", ";
        os << s.fields[i].first << " AS o" << i;
    }
    os << " FROM " << base;
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
        case PlanKind::AsofJoin: return render_asof(n, ctx);
        case PlanKind::Window: return render_window(n, ctx);
        case PlanKind::CompressedScan: return render_compressed_scan(n, ctx);
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
