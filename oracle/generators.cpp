//  WP-3 (seed of WP-9): seeded schema/data/query generators. See generators.h.
#include "oracle/generators.h"

#include <cstdint>
#include <string>
#include <vector>

#include "core/owned_batch.h"
#include "expr/expr.h"

namespace qe::oracle {
namespace {

using namespace qe::expr;

bool is_numeric(Type t) {
    return t == Type::I32 || t == Type::I64 || t == Type::F64;
}

// Column-category index lists derived from a schema, for expression generation.
struct Cols {
    std::vector<Type> type;          // type[i] = column i's type
    std::vector<std::uint32_t> num;  // numeric column indices
    std::vector<std::uint32_t> boolean;
    std::vector<std::uint32_t> ts;
};

Cols classify(const Schema& s) {
    Cols c;
    for (std::uint32_t i = 0; i < s.fields.size(); ++i) {
        const Type t = s.fields[i].second;
        c.type.push_back(t);
        if (is_numeric(t)) c.num.push_back(i);
        if (t == Type::BOOL) c.boolean.push_back(i);
        if (t == Type::TS) c.ts.push_back(i);
    }
    return c;
}

std::int64_t rand_in(std::mt19937_64& rng, std::int64_t lo, std::int64_t hi) {
    return std::uniform_int_distribution<std::int64_t>(lo, hi)(rng);
}

Expr num_literal(std::mt19937_64& rng) {
    switch (rng() % 3) {
        case 0:
            return lit(Scalar::i32(static_cast<std::int32_t>(
                rand_in(rng, -kI32Abs, kI32Abs))));
        case 1:
            return lit(Scalar::i64(rand_in(rng, -kI64Abs, kI64Abs)));
        default: {
            const double v =
                std::uniform_real_distribution<double>(-kF64Abs, kF64Abs)(rng);
            return lit(Scalar::f64(v));
        }
    }
}

Expr num_leaf(std::mt19937_64& rng, const Cols& c) {
    if (c.num.empty() || (rng() & 1u)) return num_literal(rng);
    const std::uint32_t idx = c.num[rng() % c.num.size()];
    return col(c.type[idx], idx);
}

// A widening cast of a numeric leaf, or the leaf unchanged when no safe widening
// applies. Widening only (never lossy/OOR) — see the divergence note.
Expr maybe_widen(std::mt19937_64& rng, Expr leaf) {
    switch (leaf.type()) {
        case Type::I32:
            return cast(leaf, (rng() & 1u) ? Type::I64 : Type::F64);
        case Type::I64:
            return cast(leaf, Type::F64);
        default:
            return leaf;  // F64 already widest numeric
    }
}

// Bounded numeric expression. depth<=2 keeps magnitudes overflow-free (see
// generators.h). Multiplication is leaf*small-literal only, also for bounding.
Expr gen_num(std::mt19937_64& rng, const Cols& c, int depth) {
    if (depth <= 0 || rng() % 3 == 0) return num_leaf(rng, c);
    switch (rng() % 4) {
        case 0:
            return add(gen_num(rng, c, depth - 1), gen_num(rng, c, depth - 1));
        case 1:
            return sub(gen_num(rng, c, depth - 1), gen_num(rng, c, depth - 1));
        case 2:
            return mul(num_leaf(rng, c),
                       lit(Scalar::i32(static_cast<std::int32_t>(
                           rand_in(rng, -kMulAbs, kMulAbs)))));
        default:
            return maybe_widen(rng, num_leaf(rng, c));
    }
}

const CmpOp kCmps[] = {CmpOp::Lt, CmpOp::Le, CmpOp::Gt,
                       CmpOp::Ge, CmpOp::Eq, CmpOp::Ne};

// A single comparison/leaf BOOL predicate.
Expr gen_pred_leaf(std::mt19937_64& rng, const Cols& c) {
    // Prefer a numeric comparison; fall back to bool col / ts comparison.
    const int choice = static_cast<int>(rng() % 4);
    if (choice == 0 && !c.boolean.empty()) {
        const std::uint32_t idx = c.boolean[rng() % c.boolean.size()];
        return col(Type::BOOL, idx);
    }
    if (choice == 1 && !c.ts.empty()) {
        const std::uint32_t idx = c.ts[rng() % c.ts.size()];
        return cmp(kCmps[rng() % 6], col(Type::TS, idx),
                   lit(Scalar::ts(rand_in(rng, -kI64Abs, kI64Abs))));
    }
    return cmp(kCmps[rng() % 6], gen_num(rng, c, 1), gen_num(rng, c, 1));
}

Expr gen_pred(std::mt19937_64& rng, const Cols& c, int depth) {
    if (depth <= 0 || rng() % 3 == 0) return gen_pred_leaf(rng, c);
    switch (rng() % 3) {
        case 0:
            return logic_and(gen_pred(rng, c, depth - 1),
                             gen_pred(rng, c, depth - 1));
        case 1:
            return logic_or(gen_pred(rng, c, depth - 1),
                            gen_pred(rng, c, depth - 1));
        default:
            return logic_not(gen_pred(rng, c, depth - 1));
    }
}

// One projection expression of arbitrary (inferred) type.
Expr gen_proj_expr(std::mt19937_64& rng, const Cols& c) {
    switch (rng() % 4) {
        case 0:  // a raw column of any type
            if (!c.type.empty()) {
                const std::uint32_t idx =
                    static_cast<std::uint32_t>(rng() % c.type.size());
                return col(c.type[idx], idx);
            }
            [[fallthrough]];
        case 1:
            return gen_num(rng, c, 2);
        case 2:
            return gen_pred(rng, c, 1);
        default:
            return num_literal(rng);
    }
}

Type rand_type(std::mt19937_64& rng) {
    switch (rng() % 5) {
        case 0: return Type::I32;
        case 1: return Type::I64;
        case 2: return Type::F64;
        case 3: return Type::BOOL;
        default: return Type::TS;
    }
}

Type rand_numeric_type(std::mt19937_64& rng) {
    switch (rng() % 3) {
        case 0: return Type::I32;
        case 1: return Type::I64;
        default: return Type::F64;
    }
}

OwnedColumn gen_column(std::mt19937_64& rng, Type t, std::size_t n,
                       int null_pct) {
    OwnedColumn c = OwnedColumn::make(t, n);
    void* d = c.mutable_data();
    for (std::size_t i = 0; i < n; ++i) {
        switch (t) {
            case Type::I32:
                static_cast<std::int32_t*>(d)[i] =
                    static_cast<std::int32_t>(rand_in(rng, -kI32Abs, kI32Abs));
                break;
            case Type::I64:
            case Type::TS:
                static_cast<std::int64_t*>(d)[i] =
                    rand_in(rng, -kI64Abs, kI64Abs);
                break;
            case Type::F64:
                static_cast<double*>(d)[i] =
                    std::uniform_real_distribution<double>(-kF64Abs,
                                                           kF64Abs)(rng);
                break;
            case Type::BOOL:
                static_cast<std::uint8_t*>(d)[i] = (rng() & 1u) ? 1 : 0;
                break;
        }
        if (null_pct > 0 && static_cast<int>(rng() % 100) < null_pct)
            c.set_null(i);
    }
    return c;
}

}  // namespace

Schema gen_schema(std::mt19937_64& rng, const GenConfig& cfg) {
    const int ncols =
        cfg.min_cols +
        static_cast<int>(rng() % static_cast<unsigned>(cfg.max_cols -
                                                       cfg.min_cols + 1));
    Schema s;
    for (int i = 0; i < ncols; ++i) {
        // Column 0 is always numeric so predicates/projections have a numeric
        // operand to work with.
        const Type t = (i == 0) ? rand_numeric_type(rng) : rand_type(rng);
        s.fields.emplace_back("c" + std::to_string(i), t);
    }
    return s;
}

Table gen_table(std::mt19937_64& rng, const Schema& schema,
                const GenConfig& cfg) {
    const std::size_t n =
        cfg.min_rows +
        static_cast<std::size_t>(
            rng() % (cfg.max_rows - cfg.min_rows + 1));
    std::vector<OwnedColumn> cols;
    cols.reserve(schema.fields.size());
    for (const auto& f : schema.fields)
        cols.push_back(gen_column(rng, f.second, n, cfg.null_pct));
    return Table(schema, std::move(cols));
}

LogicalQuery gen_query(std::mt19937_64& rng, const Schema& schema) {
    const Cols c = classify(schema);
    LogicalQuery q;
    // ~80% of queries carry a WHERE; the rest exercise the no-filter path.
    if (rng() % 5 != 0) q.filter = gen_pred(rng, c, 2);

    const int nproj = 1 + static_cast<int>(rng() % 4);
    for (int i = 0; i < nproj; ++i)
        q.projections.push_back(
            Projection{"p" + std::to_string(i), gen_proj_expr(rng, c)});
    return q;
}

}  // namespace qe::oracle
