//  WP-3 (seed of WP-9): differential runner. See oracle/differential.h.
#include "oracle/differential.h"

#include <memory>

#include "oracle/reference_oracle.h"

namespace qe::oracle {

DiffResult run_differential(const Table& table, const LogicalQuery& q,
                            const OracleFn& oracle, std::size_t batch_size) {
    std::unique_ptr<Operator> tree =
        build_engine_pipeline(table, q, batch_size);
    const ResultSet engine = drain_operator(*tree);
    const ResultSet golden = oracle(table, q);
    return compare_result_sets(engine, golden);
}

DiffResult run_vs_reference(const Table& table, const LogicalQuery& q,
                            std::size_t batch_size) {
    return run_differential(table, q, run_reference, batch_size);
}

}  // namespace qe::oracle
