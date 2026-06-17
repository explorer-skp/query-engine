//  WP-8: Physical plan nodes + dataframe-style builder API + lowering + a
//  deterministic printable form. THIS IS WP-8's PUBLIC SURFACE (frozen at
//  acceptance; WP-10 bench and Phase-2 WPs build queries through it). NO SQL
//  PARSER — the builder below IS the query surface (a deliberate scope cut).
//
//  THE SPINE. Everything is a typed column batch flowing through composable,
//  pull-based vectorized operators (ops/operator.h). A Plan is a small typed IR —
//  one node per frozen operator (Scan/Filter/Project/Aggregate/Join/Sort) — that
//  (1) knows its OUTPUT SCHEMA without executing (so the builder validates
//  types/columns as the tree is assembled, and print/SQL can name columns), and
//  (2) LOWERS to exactly the operator tree the per-WP tests build by hand (same
//  children, same order). The SAME Plan also drives the DuckDB oracle (rendered to
//  SQL in oracle/plan_sql.h), so one description feeds both backends — that single
//  source of truth is what makes the differential honest.
//
//  COLUMN REFERENCES (documented design — the brief lets us pick; we support
//  BOTH, split by where each is natural):
//   * EXPRESSIONS (filter predicate, projection exprs) reference columns BY INDEX
//     into the child's output schema — the frozen expr::col(Type, index) path,
//     matching the frozen operator ctors. Name resolution for an expression is
//     available the dataframe way via expr::col(builder.schema(), "name"): the
//     builder exposes the child output schema through schema().
//   * KEY LISTS (group keys, join keys, sort keys) accept a ColRef that is EITHER
//     an index OR a NAME resolved against the child schema at BUILD time — the
//     dataframe-friendly path. So `.aggregate({"region"}, {...})` and
//     `.join(scan(b), {0}, {"id"}, ...)` both work.
//
//  TABLE LIFETIME (the use-after-free hazard a builder must respect). A Scan node
//  BORROWS its Table by const pointer (mirroring ops/scan.h and
//  oracle/logical_query.cpp::build_engine_pipeline). The borrowed Table(s) must
//  OUTLIVE both the Plan and any operator tree lowered from it. Plans are cheap
//  value handles over a shared, immutable node (like expr::Expr); copying/moving a
//  Plan never moves a Table. If you generate Tables to feed a Plan, keep them in
//  stable storage (e.g. std::unique_ptr<Table>) so moving the owner does not
//  dangle the Scan's pointer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/column.h"
#include "core/types.h"
#include "expr/expr.h"
#include "ops/aggregate.h"  // AggSpec / AggFunc / agg_result_type
#include "ops/join.h"       // JoinType
#include "ops/operator.h"   // Operator
#include "ops/project.h"    // Projection
#include "ops/scan.h"       // Scan::kDefaultBatchSize
#include "ops/sort.h"       // SortKey / SortDir / NullOrder
#include "ops/table.h"      // Table
#include "tsx/asof.h"       // WP-12: AsofType (additive Phase-2 plan extension)
#include "tsx/window.h"     // WP-13: WindowMode (additive Phase-2 plan extension)
#include "tsx/compress.h"   // WP-14: CompressedTable (additive Phase-2 plan extension)

namespace qe::plan {

// One node per frozen operator. The IR is intentionally 1:1 with ops/ so lowering
// is a mechanical, defect-free transcription (and a deviation is a mutation the
// oracle catches — see plan/plan_mutants.h).
// WP-12 (additive): AsofJoin is APPENDED — the existing six enumerators keep their
// values/order byte-unchanged (verified at audit via `git diff`).
// WP-13 (additive): Window is APPENDED after AsofJoin — every existing enumerator
// keeps its value/order byte-unchanged. The enum is in-memory only and never
// serialized; nothing relies on Window's integer value (the final review linearizes
// it against any sibling Phase-2 enumerator at integration).
// WP-14 (additive): CompressedScan is APPENDED after Window — every existing
// enumerator keeps its value/order byte-unchanged. The enum is in-memory only and
// never serialized; nothing relies on CompressedScan's integer value.
enum class PlanKind {
    Scan, Filter, Project, Aggregate, Join, Sort, AsofJoin, Window,
    CompressedScan };

// A column reference for KEY LISTS (group/join/sort keys): EITHER a physical
// index into the child's output schema OR a name resolved against that schema at
// build time. Implicitly constructible from an int / unsigned / string, so call
// sites read like a dataframe: {0, "region"} mixes both freely.
struct ColRef {
    bool by_name = false;
    std::uint32_t index = 0;
    std::string name;

    ColRef(int i) : by_name(false), index(static_cast<std::uint32_t>(i)) {}
    ColRef(unsigned i) : by_name(false), index(i) {}
    ColRef(std::uint64_t i) : by_name(false), index(static_cast<std::uint32_t>(i)) {}
    ColRef(const char* n) : by_name(true), name(n) {}
    ColRef(std::string n) : by_name(true), name(std::move(n)) {}

    // Resolve to a physical column index against `schema`. Throws
    // std::invalid_argument on an out-of-range index or an unknown name.
    std::uint32_t resolve(const Schema& schema) const;
};

// A sort key for the name-friendly builder path: a ColRef plus direction and
// null-ordering. Resolved to the frozen ops/sort.h SortKey at build time. (The
// builder also accepts a std::vector<SortKey> directly for the index path.)
struct SortBy {
    ColRef col;
    SortDir dir = SortDir::Asc;
    NullOrder nulls = NullOrder::Last;
};

class Plan;  // value handle, defined below

// The concrete plan node. Public-fielded (like expr::Node) so the oracle glue can
// read it; callers never construct it directly — they use the builder. Only the
// fields relevant to `kind` are populated.
struct PlanNode {
    PlanKind kind;
    Schema out_schema;            // derived at build time (no execution)
    std::vector<Plan> children;   // 0 (Scan) / 1 (Filter/Project/Aggregate/Sort)
                                  // / 2 (Join: [0]=probe/left, [1]=build/right)

    const Table* table = nullptr;            // Scan (borrowed; must outlive plan)
    expr::Expr predicate;                    // Filter (BOOL)
    std::vector<Projection> projections;     // Project
    std::vector<std::uint32_t> group_keys;   // Aggregate (child-output indices)
    std::vector<AggSpec> aggs;               // Aggregate
    std::vector<std::uint32_t> left_keys;    // Join probe keys
    std::vector<std::uint32_t> right_keys;   // Join build keys
    JoinType join_type = JoinType::Inner;    // Join
    std::vector<SortKey> sort_keys;          // Sort

    // WP-12 (additive): AsofJoin fields. Independent of the Join fields above so
    // every existing field stays byte-unchanged. left/right partition-equality keys
    // (child-output indices), the single timestamp column per side, INNER/LEFT, and
    // an optional backward tolerance window (unset => unbounded).
    std::vector<std::uint32_t> asof_left_keys;    // AsofJoin probe equality keys
    std::vector<std::uint32_t> asof_right_keys;   // AsofJoin build equality keys
    std::uint32_t asof_left_time = 0;             // AsofJoin probe timestamp col
    std::uint32_t asof_right_time = 0;            // AsofJoin build timestamp col
    tsx::AsofType asof_type = tsx::AsofType::Inner;
    std::optional<std::int64_t> asof_tolerance;   // unset => unbounded

    // WP-13 (additive): Window fields. Independent of all existing fields so every
    // existing field stays byte-unchanged. `window_keys` are the equality partition
    // keys (child-output indices; possibly empty), `window_time` the single
    // timestamp column, `window_param` the bucket width W (Tumbling) or the PRECEDING
    // row count P (Sliding), `window_aggs` the aggregates (reusing AggSpec verbatim).
    tsx::WindowMode window_mode = tsx::WindowMode::Tumbling;
    std::vector<std::uint32_t> window_keys;       // Window partition-equality keys
    std::uint32_t window_time = 0;                // Window timestamp column
    std::int64_t window_param = 0;                // W (tumbling) / P (sliding)
    std::vector<AggSpec> window_aggs;             // Window aggregates

    // WP-14 (additive): CompressedScan fields. Independent of all existing fields.
    // `ctable` is the ENGINE source — lowering decodes from this ONLY. `ctable_ref`
    // is the INDEPENDENT decompressed/source reference Table, read by the oracle SQL
    // renderer + the plan reference ONLY (the non-circular split: engine decodes the
    // compressed bytes, the oracle loads the original source, so the differential
    // proves engine-decode == source iff the codec is truly lossless). Both borrowed;
    // both must outlive the plan and any lowered tree.
    const tsx::CompressedTable* ctable = nullptr;  // CompressedScan engine source
    const Table* ctable_ref = nullptr;             // CompressedScan oracle reference
};

// An immutable, shared, value-semantics handle to a PlanNode (mirrors expr::Expr).
// Copying a Plan is cheap and trees freely share subplans.
class Plan {
   public:
    Plan() = default;
    explicit Plan(std::shared_ptr<const PlanNode> n) : node_(std::move(n)) {}

    PlanKind kind() const { return node_->kind; }
    const Schema& output_schema() const { return node_->out_schema; }
    const PlanNode& node() const { return *node_; }
    explicit operator bool() const { return static_cast<bool>(node_); }

    // Lower this plan to a live operator tree, taking ownership of the operator
    // chain. The tree BORROWS the Table(s) referenced by Scan nodes — they (and
    // this Plan) must outlive the returned tree. Produces exactly the per-WP
    // hand-built tree: same children, same order. `batch_size` forwards to Scan.
    std::unique_ptr<Operator> lower(
        std::size_t batch_size = Scan::kDefaultBatchSize) const;

    // A deterministic, structure-revealing multi-line rendering (the root node on
    // top, children indented beneath). Pure function of the tree — stable across
    // runs. No re-parse: printing only (a parser is a non-goal).
    std::string to_string() const;

   private:
    std::shared_ptr<const PlanNode> node_;
};

// ---- the fluent dataframe-style builder ------------------------------------
//
//   plan::scan(t)
//       .filter(gt(col(Type::I32, 0), lit(Scalar::i32(0))))
//       .join(plan::scan(other), {0}, {"id"}, JoinType::Inner)
//       .aggregate({0}, {AggSpec::count_star("n"), AggSpec::sum(3, "tot")})
//       .sort({SortBy{"n", SortDir::Desc}})
//       .project({{"name", col(Type::I64, 1)}})
//       .build();   // -> std::unique_ptr<Operator>
//
// Each method validates against the current output schema and returns a NEW
// builder atop the new node (the underlying Plan is immutable). build() lowers.
class PlanBuilder {
   public:
    explicit PlanBuilder(Plan p) : plan_(std::move(p)) {}

    PlanBuilder filter(expr::Expr predicate) const;
    PlanBuilder project(std::vector<Projection> projections) const;
    PlanBuilder aggregate(std::vector<ColRef> keys,
                          std::vector<AggSpec> aggs) const;
    // `this` is the probe (left) side; `build` is the build (right) side. Output
    // columns are all left columns then all right columns (ops/join.h order).
    PlanBuilder join(const Plan& build, std::vector<ColRef> left_keys,
                     std::vector<ColRef> right_keys, JoinType type) const;
    PlanBuilder sort(std::vector<SortKey> keys) const;   // index path (frozen)
    PlanBuilder sort(std::vector<SortBy> keys) const;    // name-friendly path

    // WP-12 (additive): backward AS-OF join. `this` is the probe (left) side,
    // `build` the build (right) side. `left_keys`/`right_keys` are the equality
    // partition keys (equal length, possibly empty), `left_time`/`right_time` the
    // single ordering (timestamp) column per side. `type` selects INNER vs LEFT;
    // `tolerance` (optional) bounds the backward window (t - tb <= tolerance, in the
    // timestamp column's integer units). Output columns are all left columns then
    // all right columns (tsx/asof.h order, same as join()).
    PlanBuilder asof_join(const Plan& build, std::vector<ColRef> left_keys,
                          std::vector<ColRef> right_keys, ColRef left_time,
                          ColRef right_time, tsx::AsofType type,
                          std::optional<std::int64_t> tolerance =
                              std::nullopt) const;

    // WP-13 (additive): windowed / time-bucketed aggregation. Both builders resolve
    // their `keys`/`time` ColRefs against the child schema at build time (like
    // aggregate()), validate the timestamp column is TS/I32/I64, and derive the
    // output schema WITHOUT executing.
    //  * window_tumbling: GROUP BY (keys…, t/`width` bucket); `width` (W) > 0. Output
    //    = [keys…, bucket_start(time type), agg result columns…].
    //  * window_sliding: running aggregate over ROWS BETWEEN `preceding` (P >= 0)
    //    PRECEDING AND CURRENT ROW, partitioned by `keys`, ordered by `time`. Output
    //    = [all child columns…, agg result columns…] (one row per input row).
    PlanBuilder window_tumbling(std::vector<ColRef> keys, ColRef time,
                                std::int64_t width,
                                std::vector<AggSpec> aggs) const;
    PlanBuilder window_sliding(std::vector<ColRef> keys, ColRef time,
                               std::int64_t preceding,
                               std::vector<AggSpec> aggs) const;

    // The assembled plan (the single source of truth feeding engine + oracle).
    const Plan& plan() const { return plan_; }
    // Implicit so `.join(plan::scan(other), ...)` accepts a builder where a Plan
    // is wanted. Returns a reference into this builder.
    operator const Plan&() const { return plan_; }
    // The current output schema — lets a caller name-resolve an expression column
    // the dataframe way: expr::col(b.schema(), "c0").
    const Schema& schema() const { return plan_.output_schema(); }

    std::unique_ptr<Operator> build(
        std::size_t batch_size = Scan::kDefaultBatchSize) const {
        return plan_.lower(batch_size);
    }

   private:
    Plan plan_;
};

// Leaf: scan `table` (BORROWED — it must outlive the plan and any lowered tree).
PlanBuilder scan(const Table& table);

// WP-14 (additive) leaf: scan a COMPRESSED table, decoding batch-by-batch. `ct` is
// the engine source — lowering builds a tsx::CompressedScan over it and decodes ONLY
// its compressed bytes. `reference` is the INDEPENDENT decompressed/source Table the
// oracle (DuckDB SQL + the plan reference) loads instead — never the engine's own
// decode output (that would be circular). The node's output schema is ct.schema().
// BOTH `ct` and `reference` are BORROWED and must outlive the plan and any lowered
// tree. (See PlanNode::ctable / ctable_ref for the non-circularity contract.)
PlanBuilder compressed_scan(const tsx::CompressedTable& ct, const Table& reference);

}  // namespace qe::plan
