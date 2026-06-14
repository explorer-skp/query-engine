//  WP-3 (seed of WP-9), extended at WP-5: engine pipeline assembly + the shared
//  output-schema helper. See oracle/logical_query.h.
#include "oracle/logical_query.h"

#include <memory>
#include <utility>

#include "ops/aggregate.h"
#include "ops/filter.h"
#include "ops/join.h"
#include "ops/scan.h"

namespace qe::oracle {

Schema query_output_schema(const Schema& input, const LogicalQuery& q) {
    Schema s;
    if (q.has_group_by()) {
        const GroupBy& gb = *q.group_by;
        s.fields.reserve(gb.keys.size() + gb.aggs.size());
        for (std::uint32_t kc : gb.keys)
            s.fields.emplace_back(input.fields[kc].first,
                                  input.fields[kc].second);
        for (const auto& a : gb.aggs) {
            const Type in = (a.func == AggFunc::CountStar)
                                ? Type::I64
                                : input.fields[a.input_col].second;
            s.fields.emplace_back(a.out_name, agg_result_type(a.func, in));
        }
        return s;
    }
    s.fields.reserve(q.projections.size());
    for (const auto& p : q.projections)
        s.fields.emplace_back(p.name, p.expr.type());
    return s;
}

std::unique_ptr<Operator> build_engine_pipeline(const Table& table,
                                                const LogicalQuery& q,
                                                std::size_t batch_size) {
    std::unique_ptr<Operator> node =
        std::make_unique<Scan>(table, batch_size);
    if (q.has_filter())
        node = std::make_unique<Filter>(std::move(node), q.filter);
    if (q.has_group_by()) {
        const GroupBy& gb = *q.group_by;
        node = std::make_unique<Aggregate>(std::move(node), gb.keys, gb.aggs);
    } else {
        node = std::make_unique<Project>(std::move(node), q.projections);
    }
    // WP-7: ORDER BY is a top-level modifier applied AFTER project/aggregate; its
    // keys reference the query's OUTPUT columns, which are exactly this node's
    // output columns.
    if (q.has_order_by())
        node = std::make_unique<Sort>(std::move(node), *q.order_by);
    return node;
}

Schema join_output_schema(const Schema& probe, const Schema& build) {
    Schema s;
    s.fields.reserve(probe.fields.size() + build.fields.size());
    for (const auto& f : probe.fields) s.fields.push_back(f);
    for (const auto& f : build.fields) s.fields.push_back(f);
    return s;
}

std::unique_ptr<Operator> build_join_pipeline(const Table& probe,
                                              const Table& build,
                                              const JoinQuery& jq,
                                              std::size_t batch_size) {
    auto probe_scan = std::make_unique<Scan>(probe, batch_size);
    auto build_scan = std::make_unique<Scan>(build, batch_size);
    return std::make_unique<HashJoin>(std::move(probe_scan),
                                      std::move(build_scan), jq.probe_keys,
                                      jq.build_keys, jq.type);
}

}  // namespace qe::oracle
