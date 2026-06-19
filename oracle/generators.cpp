//  WP-3 (seed of WP-9): seeded schema/data/query generators. See generators.h.
#include "oracle/generators.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
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
        case Type::STR:
            break;  // WP-7b: STR keys are not generated (rand_key_type excludes it)
    }
}

// WP-7b: a small fixed vocabulary of short ASCII strings (no embedded NULs), the
// divergence-safe alphabet for generated VARCHAR data (generators.h philosophy).
const std::vector<std::string>& default_str_alphabet() {
    static const std::vector<std::string> a = {
        "a",    "bb",   "cat",  "dog",   "echo", "fox",
        "gamma", "hi",  "ix",   "joy",   "kilo", "lima"};
    return a;
}

OwnedColumn gen_column(std::mt19937_64& rng, Type t, std::size_t n,
                       int null_pct) {
    if (t == Type::STR)
        return gen_string_column(rng, default_str_alphabet(), n, null_pct);
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
            case Type::STR:
                break;  // handled above (early return)
        }
        if (null_pct > 0 && static_cast<int>(rng() % 100) < null_pct)
            c.set_null(i);
    }
    return c;
}

}  // namespace

OwnedColumn gen_string_column(std::mt19937_64& rng,
                              const std::vector<std::string>& alphabet,
                              std::size_t n, int null_pct) {
    // A FRESH per-column dict: two columns built from the same alphabet get the
    // same string VALUES but generally DIFFERENT codes (codes are assigned in
    // first-use order, which varies with the random pick order) — exactly the
    // cross-dictionary case STR equality/join must handle by VALUE, not by code.
    auto dict = std::make_shared<StringDict>();
    OwnedColumn c = OwnedColumn::make_str(n, dict);
    auto* codes = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& v = alphabet[rng() % alphabet.size()];
        codes[i] = dict->intern(v);
        if (null_pct > 0 && static_cast<int>(rng() % 100) < null_pct)
            c.set_null(i);
    }
    return c;
}

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

namespace {

// A timestamp column type for the as-of ordering column (TS most often; also I64 /
// I32 to exercise the integer-timestamp path the contract allows).
Type rand_time_type(std::mt19937_64& rng) {
    switch (rng() % 4) {
        case 0: return Type::I32;
        case 1: return Type::I64;
        default: return Type::TS;  // bias toward TS (the headline type)
    }
}

// Write timestamp `v` (int64) into time column `c` of type `t` at row `i`.
void put_time(OwnedColumn& c, Type t, std::size_t i, std::int64_t v) {
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
        case Type::BOOL:
        case Type::STR:
            break;  // not a timestamp type (unreachable for asof)
    }
}

// `count` GLOBALLY DISTINCT timestamps within a bounded range (so build (key,ts) is
// unique => deterministic nearest-preceding). Sampled without replacement, then
// LEFT IN RANDOM ORDER (not pre-sorted) so the operator's own sort is exercised.
std::vector<std::int64_t> distinct_timestamps(std::mt19937_64& rng,
                                              std::size_t count,
                                              std::int64_t lo, std::int64_t hi) {
    std::vector<std::int64_t> out;
    if (count == 0) return out;
    std::set<std::int64_t> seen;
    // Range is far larger than count, so rejection sampling terminates quickly.
    while (out.size() < count) {
        const std::int64_t v = rand_in(rng, lo, hi);
        if (seen.insert(v).second) out.push_back(v);
    }
    return out;
}

}  // namespace

AsofCase gen_asof_case(std::mt19937_64& rng) {
    // 0 (global), 1 (single), or 2 (composite) partition keys.
    const int nk = static_cast<int>(rng() % 3);
    std::vector<Type> kt(static_cast<std::size_t>(nk));
    for (int j = 0; j < nk; ++j) kt[static_cast<std::size_t>(j)] = rand_key_type(rng);
    const Type ttype = rand_time_type(rng);
    // Timestamp range: well within I32 magnitude so an I32 time column is safe; the
    // window is far wider than the row counts, giving irregular gaps.
    const std::int64_t TLO = -20000, THI = 20000;

    const int D = 1 + static_cast<int>(rng() % 8);  // key cardinality 1..8
    std::vector<std::vector<std::int64_t>> dom(
        static_cast<std::size_t>(nk), std::vector<std::int64_t>(D));
    for (int j = 0; j < nk; ++j)
        for (int t = 0; t < D; ++t) dom[j][t] = rand_in(rng, 0, 99);

    const int match_pct = 50 + static_cast<int>(rng() % 51);  // 50..100% in-domain
    // NOTE: NULL keys and NULL timestamps are deliberately NOT generated. Two
    // verified DuckDB v1.1.3 ASOF divergences make a byte-for-byte differential on
    // them impossible or flaky, so — exactly like the div-by-zero / overflow
    // divergences in this file — we CONSTRAIN GENERATION and validate the engine's
    // own (principled "NULL never matches") semantics against the independent
    // reference instead (see the asof_differential_test NULL edge cases):
    //   (1) NULL TIMESTAMP: DuckDB matches a NULL probe timestamp to a NULL build
    //       timestamp within a partition (NULL >= NULL treated as a self-equal
    //       group); the engine takes the principled never-match path.
    //   (2) NULL KEY: DuckDB's ASOF NULL-key matching is DATA-DEPENDENT — a probe
    //       with a NULL key component matches a NULL-key build row when the full
    //       table is present, but NOT when that probe row is queried in isolation
    //       (a hash-partition artifact). There is no well-defined semantics to
    //       mirror, so we exclude it. (A NULL key correctly never matches in a
    //       regular hash join — WP-6 — which is NOT data-dependent.)
    const int keynull_pct = 0;  // see note above (engine: NULL key never matches)
    const std::size_t B = static_cast<std::size_t>(rng() % 251);  // build rows
    const std::size_t P = static_cast<std::size_t>(rng() % 251);  // probe rows
    const int npb = static_cast<int>(rng() % 3);  // 0..2 build payload cols
    const int npp = static_cast<int>(rng() % 3);  // 0..2 probe payload cols

    // Schemas: c0..c{nk-1} keys, c{nk} timestamp, then payloads.
    Schema bs, ps;
    for (int j = 0; j < nk; ++j)
        bs.fields.emplace_back("c" + std::to_string(j), kt[static_cast<std::size_t>(j)]);
    bs.fields.emplace_back("c" + std::to_string(nk), ttype);
    for (int j = 0; j < nk; ++j)
        ps.fields.emplace_back("c" + std::to_string(j), kt[static_cast<std::size_t>(j)]);
    ps.fields.emplace_back("c" + std::to_string(nk), ttype);
    std::vector<Type> bp(static_cast<std::size_t>(npb)), pp(static_cast<std::size_t>(npp));
    for (int j = 0; j < npb; ++j) {
        bp[static_cast<std::size_t>(j)] = rand_type(rng);
        bs.fields.emplace_back("c" + std::to_string(nk + 1 + j), bp[static_cast<std::size_t>(j)]);
    }
    for (int j = 0; j < npp; ++j) {
        pp[static_cast<std::size_t>(j)] = rand_type(rng);
        ps.fields.emplace_back("c" + std::to_string(nk + 1 + j), pp[static_cast<std::size_t>(j)]);
    }

    // ---- BUILD side: keys from the domain; globally-distinct timestamps so
    // (key, ts) is unique; a keynull/tsnull fraction nulls a key col / the ts. ----
    std::vector<OwnedColumn> bkeys(static_cast<std::size_t>(nk));
    for (int j = 0; j < nk; ++j) bkeys[static_cast<std::size_t>(j)] = OwnedColumn::make(kt[static_cast<std::size_t>(j)], B);
    OwnedColumn btime = OwnedColumn::make(ttype, B);
    const std::vector<std::int64_t> bts = distinct_timestamps(rng, B, TLO, THI);
    for (std::size_t i = 0; i < B; ++i) {
        const bool null_key =
            keynull_pct > 0 && static_cast<int>(rng() % 100) < keynull_pct;
        const int null_col = (null_key && nk > 0) ? static_cast<int>(rng() % nk) : -1;
        const int t = static_cast<int>(rng() % static_cast<unsigned>(D));
        for (int j = 0; j < nk; ++j) {
            put_key(bkeys[static_cast<std::size_t>(j)], kt[static_cast<std::size_t>(j)], i, dom[static_cast<std::size_t>(j)][t]);
            if (j == null_col) bkeys[static_cast<std::size_t>(j)].set_null(i);
        }
        put_time(btime, ttype, i, bts[i]);  // never NULL (see note above)
    }
    std::vector<OwnedColumn> bcols;
    for (int j = 0; j < nk; ++j) bcols.push_back(std::move(bkeys[static_cast<std::size_t>(j)]));
    bcols.push_back(std::move(btime));
    for (int j = 0; j < npb; ++j) bcols.push_back(gen_column(rng, bp[static_cast<std::size_t>(j)], B, 15));
    Table build(bs, std::move(bcols));

    // ---- PROBE side: match_pct draw an in-domain key; timestamps are irregular
    // and SOMETIMES exactly equal a build timestamp (head-on `>=` boundary test).
    std::vector<OwnedColumn> pkeys(static_cast<std::size_t>(nk));
    for (int j = 0; j < nk; ++j) pkeys[static_cast<std::size_t>(j)] = OwnedColumn::make(kt[static_cast<std::size_t>(j)], P);
    OwnedColumn ptime = OwnedColumn::make(ttype, P);
    for (std::size_t i = 0; i < P; ++i) {
        const bool null_key =
            keynull_pct > 0 && static_cast<int>(rng() % 100) < keynull_pct;
        const bool in_domain = static_cast<int>(rng() % 100) < match_pct;
        const int null_col = (null_key && nk > 0) ? static_cast<int>(rng() % nk) : -1;
        const int t = static_cast<int>(rng() % static_cast<unsigned>(D));
        for (int j = 0; j < nk; ++j) {
            const std::int64_t v = in_domain ? dom[static_cast<std::size_t>(j)][t]
                                             : rand_in(rng, 1000, 2000);
            put_key(pkeys[static_cast<std::size_t>(j)], kt[static_cast<std::size_t>(j)], i, v);
            if (j == null_col) pkeys[static_cast<std::size_t>(j)].set_null(i);
        }
        // ~1/3 of probe timestamps land EXACTLY on a build timestamp (boundary);
        // the rest are free (gaps, before-all, after-all).
        std::int64_t pv;
        if (!bts.empty() && rng() % 3 == 0)
            pv = bts[rng() % bts.size()];
        else
            pv = rand_in(rng, TLO - 5000, THI + 5000);
        put_time(ptime, ttype, i, pv);  // never NULL (see note above)
    }
    std::vector<OwnedColumn> pcols;
    for (int j = 0; j < nk; ++j) pcols.push_back(std::move(pkeys[static_cast<std::size_t>(j)]));
    pcols.push_back(std::move(ptime));
    for (int j = 0; j < npp; ++j) pcols.push_back(gen_column(rng, pp[static_cast<std::size_t>(j)], P, 15));
    Table probe(ps, std::move(pcols));

    AsofCase c{std::move(probe), std::move(build), {}, {}, 0, 0,
               qe::tsx::AsofType::Inner, std::nullopt};
    for (int j = 0; j < nk; ++j) {
        c.left_keys.push_back(static_cast<std::uint32_t>(j));
        c.right_keys.push_back(static_cast<std::uint32_t>(j));
    }
    c.left_time = static_cast<std::uint32_t>(nk);
    c.right_time = static_cast<std::uint32_t>(nk);
    c.type = (rng() % 2) == 0 ? qe::tsx::AsofType::Inner : qe::tsx::AsofType::Left;
    // ~40% carry a tolerance window (a few timestamp units up to a wide span).
    if (rng() % 5 < 2)
        c.tolerance = static_cast<std::int64_t>(rng() % 8000);
    return c;
}

WindowCase gen_window_case(std::mt19937_64& rng) {
    const bool tumbling = (rng() % 2) == 0;
    const int nk = static_cast<int>(rng() % 3);  // 0 / 1 / 2 partition keys
    std::vector<Type> kt(static_cast<std::size_t>(nk));
    for (int j = 0; j < nk; ++j)
        kt[static_cast<std::size_t>(j)] = rand_key_type(rng);
    const Type ttype = rand_time_type(rng);

    // Row count: usually small (crosses the 64/256 batch boundaries with a tail);
    // ~1/4 of cases are large enough to straddle a full 2048 batch; some are empty.
    std::size_t n = static_cast<std::size_t>(rng() % 512);
    if (rng() % 4 == 0) n += static_cast<std::size_t>(rng() % 4096);

    // Globally-distinct timestamps in [0, HI] satisfy BOTH constraints at once:
    // >= 0 (tumbling divergence-free) AND per-partition distinct (sliding total
    // order). HI is far wider than n so rejection sampling terminates fast and
    // fits I32 when the time column is I32.
    const std::int64_t HI =
        static_cast<std::int64_t>(16 * (n + 1)) + 256;
    const std::vector<std::int64_t> ts = distinct_timestamps(rng, n, 0, HI);

    const int D = 1 + static_cast<int>(rng() % 8);  // key cardinality 1..8
    std::vector<std::vector<std::int64_t>> dom(
        static_cast<std::size_t>(nk), std::vector<std::int64_t>(D));
    for (int j = 0; j < nk; ++j)
        for (int t = 0; t < D; ++t)
            dom[static_cast<std::size_t>(j)][t] = rand_in(rng, 0, 99);
    const int keynull_pct = static_cast<int>(rng() % 20);  // NULL partition keys
    const int npay = static_cast<int>(rng() % 3);  // 0..2 payload columns

    // Schema: c0..c{nk-1} keys, c{nk} timestamp, then payloads.
    Schema s;
    for (int j = 0; j < nk; ++j)
        s.fields.emplace_back("c" + std::to_string(j),
                              kt[static_cast<std::size_t>(j)]);
    s.fields.emplace_back("c" + std::to_string(nk), ttype);
    std::vector<Type> pay(static_cast<std::size_t>(npay));
    for (int j = 0; j < npay; ++j) {
        pay[static_cast<std::size_t>(j)] = rand_type(rng);
        s.fields.emplace_back("c" + std::to_string(nk + 1 + j),
                              pay[static_cast<std::size_t>(j)]);
    }

    // Build the columns.
    std::vector<OwnedColumn> cols;
    std::vector<OwnedColumn> kcols(static_cast<std::size_t>(nk));
    for (int j = 0; j < nk; ++j)
        kcols[static_cast<std::size_t>(j)] =
            OwnedColumn::make(kt[static_cast<std::size_t>(j)], n);
    OwnedColumn tcol = OwnedColumn::make(ttype, n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool null_key =
            keynull_pct > 0 && static_cast<int>(rng() % 100) < keynull_pct;
        const int null_col =
            (null_key && nk > 0) ? static_cast<int>(rng() % nk) : -1;
        const int t = static_cast<int>(rng() % static_cast<unsigned>(D));
        for (int j = 0; j < nk; ++j) {
            put_key(kcols[static_cast<std::size_t>(j)],
                    kt[static_cast<std::size_t>(j)], i,
                    dom[static_cast<std::size_t>(j)][t]);
            if (j == null_col) kcols[static_cast<std::size_t>(j)].set_null(i);
        }
        put_time(tcol, ttype, i, ts[i]);  // never NULL
    }
    for (int j = 0; j < nk; ++j) cols.push_back(std::move(kcols[static_cast<std::size_t>(j)]));
    cols.push_back(std::move(tcol));
    for (int j = 0; j < npay; ++j)
        cols.push_back(gen_column(rng, pay[static_cast<std::size_t>(j)], n, 15));

    WindowCase c{Table(s, std::move(cols)),
                 tumbling ? qe::tsx::WindowMode::Tumbling
                          : qe::tsx::WindowMode::Sliding,
                 {},
                 static_cast<std::uint32_t>(nk),
                 1,
                 {}};
    for (int j = 0; j < nk; ++j) c.keys.push_back(static_cast<std::uint32_t>(j));

    // Param from a small set (bucket width W > 0 / preceding row count P >= 0).
    if (tumbling) {
        static const std::int64_t kW[] = {1, 2, 5, 10, 100, 1000, 5000};
        c.param = kW[rng() % (sizeof(kW) / sizeof(kW[0]))];
    } else {
        static const std::int64_t kP[] = {0, 1, 2, 5, 10, 50};
        c.param = kP[rng() % (sizeof(kP) / sizeof(kP[0]))];
    }

    // Aggregate subset: 1..4. SUM/AVG over numeric columns only (the magnitude
    // bounds keep every per-bucket / per-frame integer sum within the I64
    // accumulator). MIN/MAX/COUNT over any column. Reuse the AggSpec vocabulary.
    const std::uint32_t ncols = static_cast<std::uint32_t>(s.fields.size());
    std::vector<std::uint32_t> numeric;
    for (std::uint32_t i = 0; i < ncols; ++i)
        if (is_numeric(s.fields[i].second)) numeric.push_back(i);
    const int na = 1 + static_cast<int>(rng() % 4);
    for (int i = 0; i < na; ++i) {
        const std::string name = "a" + std::to_string(i);
        switch (rng() % 6) {
            case 0:
                c.aggs.push_back(AggSpec::count_star(name));
                break;
            case 1:
                c.aggs.push_back(
                    AggSpec::count(static_cast<std::uint32_t>(rng() % ncols), name));
                break;
            case 2:
                if (!numeric.empty())
                    c.aggs.push_back(
                        AggSpec::sum(numeric[rng() % numeric.size()], name));
                else
                    c.aggs.push_back(AggSpec::count_star(name));
                break;
            case 3:
                if (!numeric.empty())
                    c.aggs.push_back(
                        AggSpec::avg(numeric[rng() % numeric.size()], name));
                else
                    c.aggs.push_back(AggSpec::count_star(name));
                break;
            case 4:
                c.aggs.push_back(
                    AggSpec::min(static_cast<std::uint32_t>(rng() % ncols), name));
                break;
            default:
                c.aggs.push_back(
                    AggSpec::max(static_cast<std::uint32_t>(rng() % ncols), name));
                break;
        }
    }
    return c;
}

PlanCase gen_plan_case(std::mt19937_64& rng) {
    // Two correlated tables (probe/build) sharing key column(s) 0..nk-1.
    JoinCase jc = gen_join_case(rng);

    PlanCase out;
    out.tables.push_back(std::make_unique<Table>(std::move(jc.probe)));
    out.tables.push_back(std::make_unique<Table>(std::move(jc.build)));
    const Table& probe = *out.tables[0];
    const Table& build = *out.tables[1];

    plan::PlanBuilder b = plan::scan(probe);

    // ~50% prepend a filter on probe key column 0 (always numeric/TS key type),
    // exercising filter-before-join composition.
    if (rng() % 2 == 0) {
        const Type kt = probe.schema().fields[0].second;
        Expr rhs;
        switch (kt) {
            case Type::I32: rhs = lit(Scalar::i32(50)); break;
            case Type::I64: rhs = lit(Scalar::i64(50)); break;
            case Type::F64: rhs = lit(Scalar::f64(50.0)); break;
            case Type::TS:  rhs = lit(Scalar::ts(50)); break;
            case Type::BOOL:
            case Type::STR:  rhs = lit(Scalar::i32(50)); break;  // unreachable
        }
        b = b.filter(cmp(CmpOp::Lt, col(kt, 0), std::move(rhs)));
    }

    // Join probe |x| build on the generated equi-keys.
    std::vector<plan::ColRef> lk, rk;
    for (auto i : jc.query.probe_keys)
        lk.push_back(plan::ColRef(static_cast<int>(i)));
    for (auto i : jc.query.build_keys)
        rk.push_back(plan::ColRef(static_cast<int>(i)));
    b = b.join(plan::scan(build), lk, rk, jc.query.type);

    // Aggregate over the join output: GROUP BY the (probe) key column 0, with only
    // overflow-proof aggregates (COUNT(*)/COUNT/MIN/MAX) so the diff is exact.
    const std::uint32_t njoin =
        static_cast<std::uint32_t>(b.schema().fields.size());
    std::vector<AggSpec> aggs;
    aggs.push_back(AggSpec::count_star("a0"));
    aggs.push_back(AggSpec::min(0, "a1"));
    aggs.push_back(AggSpec::max(0, "a2"));
    aggs.push_back(AggSpec::count(
        static_cast<std::uint32_t>(rng() % njoin), "a3"));
    b = b.aggregate({0}, std::move(aggs));

    // ORDER BY every output column ASC NULLS LAST: groups are distinct by the key,
    // so column 0 alone already makes the order TOTAL; appending the rest is
    // belt-and-suspenders for positional compare.
    std::vector<SortKey> sk;
    const std::uint32_t nout =
        static_cast<std::uint32_t>(b.schema().fields.size());
    for (std::uint32_t i = 0; i < nout; ++i)
        sk.push_back(SortKey{i, SortDir::Asc, NullOrder::Last});
    b = b.sort(std::move(sk));

    out.plan = b.plan();
    return out;
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
