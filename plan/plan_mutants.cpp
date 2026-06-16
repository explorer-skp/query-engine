//  WP-8 mutation self-test support. See plan/plan_mutants.h.
#include "plan/plan_mutants.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ops/aggregate.h"
#include "ops/filter.h"
#include "ops/join.h"
#include "ops/project.h"
#include "ops/sort.h"
#include "tsx/asof.h"  // WP-12: faithful AsofJoin lowering (no planted defect here)

namespace qe::plan {

std::unique_ptr<Operator> lower_mutant(const Plan& p, LowerMutation m,
                                       std::size_t bs) {
    const PlanNode& n = p.node();
    switch (n.kind) {
        case PlanKind::Scan:
            return std::make_unique<Scan>(*n.table, bs);
        case PlanKind::Filter:
            if (m == LowerMutation::kDropFilter)
                return lower_mutant(n.children[0], m, bs);  // BUG: drop Filter
            return std::make_unique<Filter>(lower_mutant(n.children[0], m, bs),
                                            n.predicate);
        case PlanKind::Project:
            return std::make_unique<Project>(lower_mutant(n.children[0], m, bs),
                                             n.projections);
        case PlanKind::Aggregate: {
            std::vector<std::uint32_t> keys = n.group_keys;
            if (m == LowerMutation::kReverseAggKeys)
                std::reverse(keys.begin(), keys.end());  // BUG: wrong key order
            return std::make_unique<Aggregate>(lower_mutant(n.children[0], m, bs),
                                               std::move(keys), n.aggs);
        }
        case PlanKind::Join:
            if (m == LowerMutation::kSwapJoinSides)
                // BUG: swap probe/build sides and their keys.
                return std::make_unique<HashJoin>(
                    lower_mutant(n.children[1], m, bs),
                    lower_mutant(n.children[0], m, bs), n.right_keys,
                    n.left_keys, n.join_type);
            return std::make_unique<HashJoin>(lower_mutant(n.children[0], m, bs),
                                              lower_mutant(n.children[1], m, bs),
                                              n.left_keys, n.right_keys,
                                              n.join_type);
        case PlanKind::Sort:
            if (m == LowerMutation::kDropSort)
                return lower_mutant(n.children[0], m, bs);  // BUG: drop Sort
            return std::make_unique<Sort>(lower_mutant(n.children[0], m, bs),
                                          n.sort_keys);
        case PlanKind::AsofJoin:
            // WP-12 (additive arm): no lowering mutation is catalogued for asof
            // (its planted defects live in tsx/asof_mutants.*); lower faithfully so
            // an asof node nested in a mutated plan still lowers correctly.
            return std::make_unique<tsx::AsofJoin>(
                lower_mutant(n.children[0], m, bs),
                lower_mutant(n.children[1], m, bs), n.asof_left_keys,
                n.asof_right_keys, n.asof_left_time, n.asof_right_time,
                n.asof_type, n.asof_tolerance);
    }
    throw std::logic_error("lower_mutant: unhandled PlanKind");
}

}  // namespace qe::plan
