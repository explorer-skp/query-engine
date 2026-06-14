//  WP-3 (seed of WP-9): seeded schema/data/query generators. See generators.h.
#include "oracle/generators.h"

#include <algorithm>
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

// WP-6: a key type for a join key column. BOOL is excluded (its 2-value domain
// makes for degenerate cardinality); F64 keys carry integer-valued doubles so
// equality is exact across the engine / reference / DuckDB.
Type rand_key_type(std::mt19937_64& rng) {
    switch (rng() % 4) {
        case 0: return Type::I32;
        case 1: return Type::I64;
        case 2: return Type::F64;
        default: return Type::TS;
    }
}

// Write integer-valued key `v` into key column `c` (type `t`) at row `i`.
void put_key(OwnedColumn& c, Type t, std::size_t i, std::int64_t v) {
    std::byte* d = c.mutable_data();
    switch (t) {
        case Type::I32:
            reinterpret_cast<std::int32_t*>(d)[i] = static_cast<std::int32_t>(v);
            break;
        case Type::I64:
        case Type::TS:
            reinterpret_cast<std::int64_t*>(d)[i] = v;
            break;
        case Type::F64:
            reinterpret_cast<double*>(d)[i] = static_cast<double>(v);
            break;
        case Type::BOOL:
            reinterpret_cast<std::uint8_t*>(d)[i] =
                static_cast<std::uint8_t>(v & 1);
            break;
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

LogicalQuery gen_group_by_query(std::mt19937_64& rng, const Schema& schema) {
    const Cols c = classify(schema);
    const std::uint32_t ncols = static_cast<std::uint32_t>(schema.fields.size());
    LogicalQuery q;
    // ~50% carry a WHERE (exercise both the filtered and unfiltered group build).
    if (rng() % 2 == 0) q.filter = gen_pred(rng, c, 2);

    GroupBy gb;
    // Key subset: 0..min(3, ncols) DISTINCT columns. 0 keys => global aggregate.
    std::vector<std::uint32_t> perm(ncols);
    for (std::uint32_t i = 0; i < ncols; ++i) perm[i] = i;
    for (std::uint32_t i = ncols; i > 1; --i)
        std::swap(perm[i - 1], perm[rng() % i]);  // Fisher-Yates (seeded)
    const std::uint32_t max_keys =
        std::min<std::uint32_t>(3, ncols);
    const std::uint32_t nk =
        static_cast<std::uint32_t>(rng() % (max_keys + 1));
    for (std::uint32_t i = 0; i < nk; ++i) gb.keys.push_back(perm[i]);

    // Aggregate subset: 1..4. SUM/AVG only over numeric columns.
    const int na = 1 + static_cast<int>(rng() % 4);
    for (int i = 0; i < na; ++i) {
        const std::string name = "a" + std::to_string(i);
        switch (rng() % 6) {
            case 0:
                gb.aggs.push_back(AggSpec::count_star(name));
                break;
            case 1:
                gb.aggs.push_back(AggSpec::count(
                    static_cast<std::uint32_t>(rng() % ncols), name));
                break;
            case 2:
                if (!c.num.empty())
                    gb.aggs.push_back(
                        AggSpec::sum(c.num[rng() % c.num.size()], name));
                else
                    gb.aggs.push_back(AggSpec::count_star(name));
                break;
            case 3:
                if (!c.num.empty())
                    gb.aggs.push_back(
                        AggSpec::avg(c.num[rng() % c.num.size()], name));
                else
                    gb.aggs.push_back(AggSpec::count_star(name));
                break;
            case 4:
                gb.aggs.push_back(AggSpec::min(
                    static_cast<std::uint32_t>(rng() % ncols), name));
                break;
            default:
                gb.aggs.push_back(AggSpec::max(
                    static_cast<std::uint32_t>(rng() % ncols), name));
                break;
        }
    }
    q.group_by = std::move(gb);
    return q;
}

JoinCase gen_join_case(std::mt19937_64& rng) {
    const int nk = 1 + static_cast<int>(rng() % 2);  // 1 or 2 keys
    std::vector<Type> kt(nk);
    for (int j = 0; j < nk; ++j) kt[j] = rand_key_type(rng);

    const int D = 1 + static_cast<int>(rng() % 12);  // key cardinality 1..12
    // Shared build-key DOMAIN, values in [0,99] (probe out-of-domain uses
    // [1000,2000], disjoint, so a non-matching probe key provably misses).
    std::vector<std::vector<std::int64_t>> dom(
        nk, std::vector<std::int64_t>(D));
    for (int j = 0; j < nk; ++j)
        for (int t = 0; t < D; ++t) dom[j][t] = rand_in(rng, 0, 99);

    const bool hot = (rng() % 2) == 0;                      // skew toward idx 0
    const int match_pct = static_cast<int>(rng() % 101);    // probe match rate
    const int keynull_pct = static_cast<int>(rng() % 30);   // NULL-key fraction
    const std::size_t B = static_cast<std::size_t>(rng() % 401);  // build rows
    const std::size_t P = static_cast<std::size_t>(rng() % 401);  // probe rows
    const int npb = static_cast<int>(rng() % 3);  // 0..2 build payload cols
    const int npp = static_cast<int>(rng() % 3);  // 0..2 probe payload cols

    Schema bs, ps;
    for (int j = 0; j < nk; ++j)
        bs.fields.emplace_back("c" + std::to_string(j), kt[j]);
    for (int j = 0; j < nk; ++j)
        ps.fields.emplace_back("c" + std::to_string(j), kt[j]);
    std::vector<Type> bp(npb), pp(npp);
    for (int j = 0; j < npb; ++j) {
        bp[j] = rand_type(rng);
        bs.fields.emplace_back("c" + std::to_string(nk + j), bp[j]);
    }
    for (int j = 0; j < npp; ++j) {
        pp[j] = rand_type(rng);
        ps.fields.emplace_back("c" + std::to_string(nk + j), pp[j]);
    }

    auto pick_index = [&]() -> int {
        if (hot && (rng() % 2) == 0) return 0;  // hot key (skew)
        return static_cast<int>(rng() % static_cast<unsigned>(D));
    };

    // BUILD side: keys drawn from the domain; a keynull_pct fraction nulls ONE
    // random key column (kNeverMatch => that row joins nothing).
    std::vector<OwnedColumn> bkeys(nk);
    for (int j = 0; j < nk; ++j) bkeys[j] = OwnedColumn::make(kt[j], B);
    for (std::size_t i = 0; i < B; ++i) {
        const bool null_key =
            keynull_pct > 0 && static_cast<int>(rng() % 100) < keynull_pct;
        const int null_col = null_key ? static_cast<int>(rng() % nk) : -1;
        const int t = pick_index();
        for (int j = 0; j < nk; ++j) {
            put_key(bkeys[j], kt[j], i, dom[j][t]);
            if (j == null_col) bkeys[j].set_null(i);
        }
    }
    std::vector<OwnedColumn> bcols;
    for (int j = 0; j < nk; ++j) bcols.push_back(std::move(bkeys[j]));
    for (int j = 0; j < npb; ++j) bcols.push_back(gen_column(rng, bp[j], B, 15));
    Table build(bs, std::move(bcols));

    // PROBE side: match_pct of rows draw an in-domain key (potential match), the
    // rest draw an out-of-domain key (guaranteed miss); keynull_pct nulls a key.
    std::vector<OwnedColumn> pkeys(nk);
    for (int j = 0; j < nk; ++j) pkeys[j] = OwnedColumn::make(kt[j], P);
    for (std::size_t i = 0; i < P; ++i) {
        const bool null_key =
            keynull_pct > 0 && static_cast<int>(rng() % 100) < keynull_pct;
        const bool in_domain = static_cast<int>(rng() % 100) < match_pct;
        const int null_col = null_key ? static_cast<int>(rng() % nk) : -1;
        const int t = pick_index();
        for (int j = 0; j < nk; ++j) {
            const std::int64_t v =
                in_domain ? dom[j][t] : rand_in(rng, 1000, 2000);
            put_key(pkeys[j], kt[j], i, v);
            if (j == null_col) pkeys[j].set_null(i);
        }
    }
    std::vector<OwnedColumn> pcols;
    for (int j = 0; j < nk; ++j) pcols.push_back(std::move(pkeys[j]));
    for (int j = 0; j < npp; ++j) pcols.push_back(gen_column(rng, pp[j], P, 15));
    Table probe(ps, std::move(pcols));

    JoinQuery q;
    for (int j = 0; j < nk; ++j) {
        q.probe_keys.push_back(static_cast<std::uint32_t>(j));
        q.build_keys.push_back(static_cast<std::uint32_t>(j));
    }
    q.type = (rng() % 2) == 0 ? JoinType::Inner : JoinType::Left;

    return JoinCase{std::move(probe), std::move(build), std::move(q)};
}

LogicalQuery gen_order_by_query(std::mt19937_64& rng, const Schema& schema) {
    const Cols c = classify(schema);
    const std::uint32_t ncols = static_cast<std::uint32_t>(schema.fields.size());
    LogicalQuery q;
    // ~50% carry a WHERE (exercise the filtered and unfiltered sort input).
    if (rng() % 2 == 0) q.filter = gen_pred(rng, c, 2);

    // Project EVERY column as-is: output schema == input schema, so the ORDER BY
    // keys (output indices) cover all column types.
    for (std::uint32_t i = 0; i < ncols; ++i)
        q.projections.push_back(
            Projection{"p" + std::to_string(i), col(c.type[i], i)});

    // Random permutation of the columns (seeded Fisher-Yates).
    std::vector<std::uint32_t> perm(ncols);
    for (std::uint32_t i = 0; i < ncols; ++i) perm[i] = i;
    for (std::uint32_t i = ncols; i > 1; --i)
        std::swap(perm[i - 1], perm[rng() % i]);

    // The first 1..ncols keys are "interesting" (random dir + null order); the
    // rest are deterministic tiebreakers (ASC NULLS LAST) so the order is TOTAL
    // and the positional differential is unambiguous (see generators.h / report).
    const std::uint32_t nprimary =
        ncols == 0 ? 0 : 1 + static_cast<std::uint32_t>(rng() % ncols);
    std::vector<SortKey> keys;
    keys.reserve(ncols);
    for (std::uint32_t i = 0; i < ncols; ++i) {
        SortKey k;
        k.col = perm[i];
        if (i < nprimary) {
            k.dir = (rng() & 1u) ? SortDir::Desc : SortDir::Asc;
            k.nulls = (rng() & 1u) ? NullOrder::First : NullOrder::Last;
        } else {
            k.dir = SortDir::Asc;
            k.nulls = NullOrder::Last;
        }
        keys.push_back(k);
    }
    q.order_by = std::move(keys);
    return q;
}

}  // namespace qe::oracle
