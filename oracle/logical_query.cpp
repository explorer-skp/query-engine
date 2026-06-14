//  WP-3 (seed of WP-9): engine pipeline assembly. See oracle/logical_query.h.
#include "oracle/logical_query.h"

#include <memory>
#include <utility>

#include "ops/filter.h"
#include "ops/scan.h"

namespace qe::oracle {

std::unique_ptr<Operator> build_engine_pipeline(const Table& table,
                                                const LogicalQuery& q,
                                                std::size_t batch_size) {
    std::unique_ptr<Operator> node =
        std::make_unique<Scan>(table, batch_size);
    if (q.has_filter())
        node = std::make_unique<Filter>(std::move(node), q.filter);
    node = std::make_unique<Project>(std::move(node), q.projections);
    return node;
}

}  // namespace qe::oracle
