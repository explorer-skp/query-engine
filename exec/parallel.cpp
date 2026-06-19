//  WP-10b: parallel-execution driver implementation. See exec/parallel.h for the
//  morsel model, the per-operator merge, the race-freedom argument, and the scope.
#include "exec/parallel.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/owned_batch.h"
#include "core/types.h"
#include "exec/morsel_scan.h"
#include "ops/aggregate.h"
#include "ops/filter.h"
#include "ops/join.h"
#include "ops/project.h"
#include "ops/scan.h"
#include "ops/sort.h"
#include "ops/table.h"

namespace qe::exec {
namespace {

using qe::oracle::Cell;
using qe::oracle::ResultSet;
using qe::plan::Plan;
using qe::plan::PlanKind;
using qe::plan::PlanNode;

constexpr std::size_t kScanBatch = 2048;  // batch size inside a morsel (mult of 64)

// ---- plan-shape predicates --------------------------------------------------

// The driving path must be morsel-izable: Scan/Filter/Project, and Join whose PROBE
// (children[0]) is itself morsel-izable (the build side is full-lowered, so it is
// unrestricted). A pipeline-breaker on the driving path is not morsel-izable.
bool streaming_morselizable(const Plan& p) {
    switch (p.kind()) {
        case PlanKind::Scan:
            return true;
        case PlanKind::Filter:
        case PlanKind::Project:
            return streaming_morselizable(p.node().children[0]);
        case PlanKind::Join:
            return streaming_morselizable(p.node().children[0]);
        default:
            return false;
    }
}

std::vector<Type> schema_types(const Schema& s) {
    std::vector<Type> t;
    t.reserve(s.fields.size());
    for (const auto& f : s.fields) t.push_back(f.second);
    return t;
}

// Descend children[0] (the driving path) to the base Scan; return its Table.
const Table* leftmost_scan_table(const Plan& p) {
    const PlanNode& n = p.node();
    if (n.kind == PlanKind::Scan) return n.table;
    if (n.children.empty()) return nullptr;
    return leftmost_scan_table(n.children[0]);
}

// ---- morsels ----------------------------------------------------------------

struct Morsel {
    std::size_t start;
    std::size_t end;
};

std::vector<Morsel> make_morsels(std::size_t total, std::size_t morsel_rows) {
    std::vector<Morsel> v;
    for (std::size_t s = 0; s < total; s += morsel_rows)
        v.push_back({s, std::min(s + morsel_rows, total)});
    if (v.empty()) v.push_back({0, 0});  // one empty morsel (global-agg / empty)
    return v;
}

// Lower `p` to a private operator tree whose DRIVING (bottom-left) Scan is a
// MorselScan over [m.start, m.end); Join build sides are full single-thread lowers.
std::unique_ptr<Operator> lower_morsel(const Plan& p, const Morsel& m) {
    const PlanNode& n = p.node();
    switch (n.kind) {
        case PlanKind::Scan:
            return std::make_unique<MorselScan>(*n.table, m.start, m.end,
                                                kScanBatch);
        case PlanKind::Filter:
            return std::make_unique<Filter>(lower_morsel(n.children[0], m),
                                            n.predicate);
        case PlanKind::Project:
            return std::make_unique<Project>(lower_morsel(n.children[0], m),
                                             n.projections);
        case PlanKind::Join:
            return std::make_unique<HashJoin>(
                lower_morsel(n.children[0], m),    // probe: morsel-ized
                n.children[1].lower(kScanBatch),   // build: full, private per worker
                n.left_keys, n.right_keys, n.join_type);
        default:
            throw std::logic_error(
                "parallel: lower_morsel reached a non-streaming node");
    }
}

// ---- ResultSet -> Table (to feed the single-thread top Sort) ----------------

Table result_set_to_table(const ResultSet& rs, const Schema& schema) {
    const std::size_t n = rs.num_rows();
    std::vector<OwnedColumn> cols;
    cols.reserve(schema.fields.size());
    for (std::size_t c = 0; c < schema.fields.size(); ++c) {
        const Type t = schema.fields[c].second;
        // WP-7b: STR intermediates re-intern their string VALUES into a fresh dict
        // owned by this rebuilt column (the codes index it).
        std::shared_ptr<StringDict> dict;
        OwnedColumn oc =
            (t == Type::STR)
                ? OwnedColumn::make_str(n, (dict = std::make_shared<StringDict>()))
                : OwnedColumn::make(t, n);
        std::byte* data = oc.mutable_data();
        for (std::size_t r = 0; r < n; ++r) {
            const Cell& cell = rs.rows[r][c];
            // Always write a value (0 for null) so no padding byte is left
            // uninitialized, then mark nulls.
            switch (t) {
                case Type::I32:
                    reinterpret_cast<std::int32_t*>(data)[r] =
                        cell.is_null ? 0 : static_cast<std::int32_t>(cell.i);
                    break;
                case Type::I64:
                case Type::TS:
                    reinterpret_cast<std::int64_t*>(data)[r] =
                        cell.is_null ? 0 : cell.i;
                    break;
                case Type::F64:
                    reinterpret_cast<double*>(data)[r] =
                        cell.is_null ? 0.0 : cell.f;
                    break;
                case Type::BOOL:
                    reinterpret_cast<std::uint8_t*>(data)[r] =
                        (!cell.is_null && cell.i) ? 1 : 0;
                    break;
                case Type::STR:
                    reinterpret_cast<std::int32_t*>(data)[r] =
                        cell.is_null ? 0 : dict->intern(cell.s);
                    break;
            }
            if (cell.is_null) oc.set_null(r);
        }
        cols.push_back(std::move(oc));
    }
    return Table(schema, std::move(cols));
}

// ---- partial-aggregate plumbing --------------------------------------------

// The kind of one PARTIAL aggregate column (post AVG-decomposition). Drives both
// how its cell is read from the per-morsel partial result and how it merges.
enum class PK { CountStar, Count, SumI, SumF, MinI, MinF, MaxI, MaxF };

// One per-group accumulator cell.
struct Acc {
    bool seen = false;
    std::int64_t i = 0;
    double f = 0.0;
};

// How to finalize one ORIGINAL aggregate from the merged accumulators.
struct OrigMap {
    AggFunc func;
    int val_idx;    // accumulator index of the value (sum/min/max/count*)
    int count_idx;  // accumulator index of the COUNT partial (AVG only; else -1)
    Type result_type;
};

bool is_f64_input(const Schema& bs, std::uint32_t col) {
    return bs.fields[col].second == Type::F64;
}

// Build the partial AggSpec list handed to each morsel's frozen Aggregate, the kind
// of each partial column, and the finalize map. AVG -> SUM + COUNT.
void build_partials(const std::vector<AggSpec>& aggs, const Schema& bs,
                    std::vector<AggSpec>& specs, std::vector<PK>& kinds,
                    std::vector<OrigMap>& omap) {
    int idx = 0;
    auto add = [&](AggSpec s, PK k) {
        specs.push_back(std::move(s));
        kinds.push_back(k);
        return idx++;
    };
    for (const AggSpec& a : aggs) {
        const std::string nm = "p" + std::to_string(idx);
        switch (a.func) {
            case AggFunc::CountStar: {
                const int v = add(AggSpec::count_star(nm), PK::CountStar);
                omap.push_back({AggFunc::CountStar, v, -1, Type::I64});
                break;
            }
            case AggFunc::Count: {
                const int v = add(AggSpec::count(a.input_col, nm), PK::Count);
                omap.push_back({AggFunc::Count, v, -1, Type::I64});
                break;
            }
            case AggFunc::Sum: {
                const bool fl = is_f64_input(bs, a.input_col);
                const int v = add(AggSpec::sum(a.input_col, nm),
                                  fl ? PK::SumF : PK::SumI);
                omap.push_back(
                    {AggFunc::Sum, v, -1, fl ? Type::F64 : Type::I64});
                break;
            }
            case AggFunc::Min: {
                const Type in = bs.fields[a.input_col].second;
                const bool fl = in == Type::F64;
                const int v = add(AggSpec::min(a.input_col, nm),
                                  fl ? PK::MinF : PK::MinI);
                omap.push_back({AggFunc::Min, v, -1, in});
                break;
            }
            case AggFunc::Max: {
                const Type in = bs.fields[a.input_col].second;
                const bool fl = in == Type::F64;
                const int v = add(AggSpec::max(a.input_col, nm),
                                  fl ? PK::MaxF : PK::MaxI);
                omap.push_back({AggFunc::Max, v, -1, in});
                break;
            }
            case AggFunc::Avg: {
                const bool fl = is_f64_input(bs, a.input_col);
                const int s = add(AggSpec::sum(a.input_col, nm),
                                  fl ? PK::SumF : PK::SumI);
                const std::string nm2 = "p" + std::to_string(idx);
                const int c = add(AggSpec::count(a.input_col, nm2), PK::Count);
                omap.push_back({AggFunc::Avg, s, c, Type::F64});
                break;
            }
        }
    }
}

// Encode the key cells of one partial row into a byte string (null flag + 8 bytes).
// The data generators emit no -0.0 / NaN, so F64 bit-pattern equality == value
// equality for generated keys (the only divergence-free regime, per generators.h).
std::string key_of(const std::vector<Cell>& row, const std::vector<Type>& kt) {
    std::string s;
    s.reserve(kt.size() * 9);
    for (std::size_t k = 0; k < kt.size(); ++k) {
        const Cell& c = row[k];
        s.push_back(c.is_null ? char(1) : char(0));
        std::uint64_t bits = 0;
        if (!c.is_null) {
            if (kt[k] == Type::F64) {
                double d = c.f;
                if (d == 0.0) d = 0.0;  // collapse -0.0 (defensive; data has none)
                std::memcpy(&bits, &d, sizeof(bits));
            } else {
                bits = static_cast<std::uint64_t>(c.i);
            }
        }
        char b[8];
        std::memcpy(b, &bits, 8);
        s.append(b, 8);
    }
    return s;
}

// One merged group: its key cells (for emit) + one accumulator per partial column.
struct Group {
    std::vector<Cell> keys;
    std::vector<Acc> acc;
};

// Fold a partial cell into an accumulator with PLAIN combine (used WITHIN a worker,
// across its morsels — never the mutated path).
void fold_plain(Acc& a, const Cell& c, PK k) {
    switch (k) {
        case PK::CountStar:
        case PK::Count:
            a.i += c.i;
            a.seen = true;
            break;
        case PK::SumI:
            if (!c.is_null) { a.i += c.i; a.seen = true; }
            break;
        case PK::SumF:
            if (!c.is_null) { a.f += c.f; a.seen = true; }
            break;
        case PK::MinI:
            if (!c.is_null) { a.i = a.seen ? std::min(a.i, c.i) : c.i; a.seen = true; }
            break;
        case PK::MinF:
            if (!c.is_null) { a.f = a.seen ? std::min(a.f, c.f) : c.f; a.seen = true; }
            break;
        case PK::MaxI:
            if (!c.is_null) { a.i = a.seen ? std::max(a.i, c.i) : c.i; a.seen = true; }
            break;
        case PK::MaxF:
            if (!c.is_null) { a.f = a.seen ? std::max(a.f, c.f) : c.f; a.seen = true; }
            break;
    }
}

}  // namespace

// ---- driver -----------------------------------------------------------------

ParallelEngine::ParallelEngine(ParallelConfig cfg) {
    unsigned t = cfg.threads;
    if (t == 0) t = std::thread::hardware_concurrency();
    if (t == 0) t = 1;  // hardware_concurrency may report 0
    threads_ = t;

    std::size_t m = cfg.morsel_rows == 0 ? kDefaultMorselRows : cfg.morsel_rows;
    if (m % 64 != 0)
        throw std::invalid_argument(
            "ParallelConfig.morsel_rows must be a multiple of 64");
    morsel_rows_ = m;
}

bool ParallelEngine::supported(const Plan& p) {
    switch (p.kind()) {
        case PlanKind::Sort:
            return supported(p.node().children[0]);
        case PlanKind::Aggregate:
            return streaming_morselizable(p.node().children[0]);
        case PlanKind::Scan:
        case PlanKind::Filter:
        case PlanKind::Project:
        case PlanKind::Join:
            return streaming_morselizable(p);
        default:
            return false;
    }
}

ResultSet ParallelEngine::run(const Plan& p) {
    if (!supported(p)) {
        // Correct single-thread fallback (documented scope cut).
        std::unique_ptr<Operator> tree = p.lower(kScanBatch);
        return qe::oracle::drain_operator(*tree);
    }
    return run_node(p);
}

ResultSet ParallelEngine::run_node(const Plan& p) {
    switch (p.kind()) {
        case PlanKind::Sort: {
            const PlanNode& n = p.node();
            const Plan& child = n.children[0];
            const ResultSet child_rs = run_node(child);
            // Single-thread final Sort over the merged input (brief-sanctioned).
            const Table tbl =
                result_set_to_table(child_rs, child.output_schema());
            Sort op(std::make_unique<Scan>(tbl, kScanBatch), n.sort_keys);
            return qe::oracle::drain_operator(op);
        }
        case PlanKind::Aggregate:
            return run_aggregate(p);
        default:
            return run_streaming(p);
    }
}

ResultSet ParallelEngine::run_streaming(const Plan& p) {
    const Table* drv = leftmost_scan_table(p);
    const std::size_t total = drv ? drv->num_rows() : 0;
    const std::vector<Morsel> morsels = make_morsels(total, morsel_rows_);

    std::vector<std::vector<ResultSet>> parts(threads_);
    std::atomic<std::size_t> cursor{0};
    auto work = [&](unsigned w) {
        std::size_t i;
        while ((i = cursor.fetch_add(1)) < morsels.size()) {
            std::unique_ptr<Operator> tree = lower_morsel(p, morsels[i]);
            parts[w].push_back(qe::oracle::drain_operator(*tree));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads_ - 1);
    for (unsigned w = 1; w < threads_; ++w) pool.emplace_back(work, w);
    work(0);
    for (std::thread& th : pool) th.join();

    // Exchange = concatenate (order-free; the diff canonicalizes for streaming).
    ResultSet out;
    out.types = schema_types(p.output_schema());
    for (const std::vector<ResultSet>& pw : parts)
        for (const ResultSet& rs : pw)
            for (const std::vector<Cell>& row : rs.rows) out.rows.push_back(row);
    return out;
}

ResultSet ParallelEngine::run_aggregate(const Plan& p) {
    const PlanNode& agg = p.node();
    const Plan& body = agg.children[0];
    const Schema& bs = body.output_schema();

    std::vector<AggSpec> specs;
    std::vector<PK> kinds;
    std::vector<OrigMap> omap;
    build_partials(agg.aggs, bs, specs, kinds, omap);

    std::vector<Type> keytypes;
    keytypes.reserve(agg.group_keys.size());
    for (std::uint32_t k : agg.group_keys) keytypes.push_back(bs.fields[k].second);
    const std::size_t nk = agg.group_keys.size();

    const Table* drv = leftmost_scan_table(body);
    const std::size_t total = drv ? drv->num_rows() : 0;
    const std::vector<Morsel> morsels = make_morsels(total, morsel_rows_);

    // Each WORKER builds a PRIVATE partial aggregate (one row per group), folding
    // its morsels with PLAIN combine. Cross-worker merge happens after join.
    std::vector<std::unordered_map<std::string, Group>> per_worker(threads_);
    std::atomic<std::size_t> cursor{0};
    auto work = [&](unsigned w) {
        std::unordered_map<std::string, Group>& map = per_worker[w];
        std::size_t mi;
        while ((mi = cursor.fetch_add(1)) < morsels.size()) {
            std::unique_ptr<Operator> tree = std::make_unique<Aggregate>(
                lower_morsel(body, morsels[mi]), agg.group_keys, specs);
            const ResultSet prs = qe::oracle::drain_operator(*tree);
            for (const std::vector<Cell>& row : prs.rows) {
                const std::string key = key_of(row, keytypes);
                auto it = map.find(key);
                if (it == map.end()) {
                    Group g;
                    g.keys.assign(row.begin(), row.begin() + nk);
                    g.acc.resize(kinds.size());
                    it = map.emplace(key, std::move(g)).first;
                }
                Group& g = it->second;
                for (std::size_t j = 0; j < kinds.size(); ++j)
                    fold_plain(g.acc[j], row[nk + j], kinds[j]);
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads_ - 1);
    for (unsigned w = 1; w < threads_; ++w) pool.emplace_back(work, w);
    work(0);
    for (std::thread& th : pool) th.join();

    // Cross-WORKER merge. The FIRST worker holding a key inserts a copy (no hook);
    // a key present in >=2 worker partials is combined via the merge_* hooks — the
    // ONLY place a >1-thread-only bug can live (the mutant overrides those hooks).
    std::unordered_map<std::string, Group> merged;
    for (std::unordered_map<std::string, Group>& wm : per_worker) {
        for (auto& kv : wm) {
            auto it = merged.find(kv.first);
            if (it == merged.end()) {
                merged.emplace(kv.first, std::move(kv.second));
                continue;
            }
            Group& d = it->second;
            const Group& s = kv.second;
            for (std::size_t j = 0; j < kinds.size(); ++j) {
                Acc& a = d.acc[j];
                const Acc& b = s.acc[j];
                switch (kinds[j]) {
                    case PK::CountStar:
                    case PK::Count:
                        a.i = merge_count(a.i, b.i);
                        a.seen = a.seen || b.seen;
                        break;
                    case PK::SumI:
                        if (b.seen) {
                            a.i = a.seen ? merge_sum_i(a.i, b.i) : b.i;
                            a.seen = true;
                        }
                        break;
                    case PK::SumF:
                        if (b.seen) {
                            a.f = a.seen ? merge_sum_f(a.f, b.f) : b.f;
                            a.seen = true;
                        }
                        break;
                    case PK::MinI:
                        if (b.seen) {
                            a.i = a.seen ? std::min(a.i, b.i) : b.i;
                            a.seen = true;
                        }
                        break;
                    case PK::MinF:
                        if (b.seen) {
                            a.f = a.seen ? std::min(a.f, b.f) : b.f;
                            a.seen = true;
                        }
                        break;
                    case PK::MaxI:
                        if (b.seen) {
                            a.i = a.seen ? std::max(a.i, b.i) : b.i;
                            a.seen = true;
                        }
                        break;
                    case PK::MaxF:
                        if (b.seen) {
                            a.f = a.seen ? std::max(a.f, b.f) : b.f;
                            a.seen = true;
                        }
                        break;
                }
            }
        }
    }

    // Finalize to the aggregate's output schema.
    ResultSet out;
    out.types = schema_types(agg.out_schema);
    out.rows.reserve(merged.size());
    for (auto& kv : merged) {
        const Group& g = kv.second;
        std::vector<Cell> row;
        row.reserve(nk + omap.size());
        for (std::size_t k = 0; k < nk; ++k) row.push_back(g.keys[k]);
        for (const OrigMap& om : omap) {
            Cell c;
            const Acc& a = g.acc[om.val_idx];
            switch (om.func) {
                case AggFunc::CountStar:
                case AggFunc::Count:
                    c.is_null = false;
                    c.i = a.i;
                    break;
                case AggFunc::Sum:
                    if (a.seen) {
                        if (om.result_type == Type::F64) c.f = a.f;
                        else c.i = a.i;
                    } else {
                        c.is_null = true;
                    }
                    break;
                case AggFunc::Min:
                case AggFunc::Max:
                    if (a.seen) {
                        if (om.result_type == Type::F64) c.f = a.f;
                        else c.i = a.i;
                    } else {
                        c.is_null = true;
                    }
                    break;
                case AggFunc::Avg: {
                    const Acc& cnt = g.acc[om.count_idx];
                    if (cnt.i > 0) {
                        const double sum =
                            kinds[om.val_idx] == PK::SumF
                                ? a.f
                                : static_cast<double>(a.i);
                        c.f = sum / static_cast<double>(cnt.i);
                    } else {
                        c.is_null = true;
                    }
                    break;
                }
            }
            row.push_back(c);
        }
        out.rows.push_back(std::move(row));
    }
    return out;
}

ResultSet run_plan_parallel(const Plan& p, ParallelConfig cfg) {
    return ParallelEngine(cfg).run(p);
}

}  // namespace qe::exec
