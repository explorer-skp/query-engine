//  WP-10b mutation self-test support. See exec/parallel_mutants.h.
#include "exec/parallel_mutants.h"

namespace qe::mutant {
namespace {

// Derives the real driver and overrides ONLY the cross-worker merge hooks: each
// returns the incoming partial (`add`) instead of accumulating, so a group split
// across >=2 worker partials keeps only the last and drops the rest. At 1 thread no
// hook fires (each group lives in one worker partial) => identical to clean.
class DropPartialEngine : public qe::exec::ParallelEngine {
   public:
    using qe::exec::ParallelEngine::ParallelEngine;

   protected:
    std::int64_t merge_count(std::int64_t /*acc*/,
                             std::int64_t add) const override {
        return add;
    }
    std::int64_t merge_sum_i(std::int64_t /*acc*/,
                             std::int64_t add) const override {
        return add;
    }
    double merge_sum_f(double /*acc*/, double add) const override {
        return add;
    }
};

}  // namespace

qe::oracle::ResultSet run_plan_parallel_mutant(const qe::plan::Plan& p,
                                               qe::exec::ParallelConfig cfg,
                                               ParallelMutation mut) {
    if (mut == ParallelMutation::kMergeDropPartial)
        return DropPartialEngine(cfg).run(p);
    return qe::exec::ParallelEngine(cfg).run(p);
}

}  // namespace qe::mutant
