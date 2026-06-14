//  WP-2: shared cast rounding + range predicate (single source of truth used by
//  BOTH cast twins AND by eval's NULL computation, so they cannot drift).
//
//  float -> integer cast semantics (aligned with DuckDB): round half away from
//  zero (std::round), then the value must be finite and land within the target
//  integer's range; otherwise the cast yields NULL (eval) and the kernels write a
//  defined dummy so no out-of-range double->int conversion (which is UB) executes.
#pragma once

#include <cmath>

#include "core/types.h"

namespace qe::expr {

// Round half away from zero (DuckDB float->int rounding).
inline double cast_round(double x) { return std::round(x); }

// True iff `x` rounds to a value representable in target integer type `to`.
// 2^31 and 2^63 are exactly representable as doubles, so these bounds are exact;
// any double passing the test converts to the integer without UB. Returns true
// for non-integer targets (they are not range-limited this way).
inline bool f64_fits_int(double x, Type to) {
    if (!std::isfinite(x)) return false;
    const double r = cast_round(x);
    switch (to) {
        case Type::I32:
            return r >= -2147483648.0 && r < 2147483648.0;  // [-2^31, 2^31)
        case Type::I64:
        case Type::TS:
            return r >= -9223372036854775808.0 &&
                   r < 9223372036854775808.0;  // [-2^63, 2^63)
        default:
            return true;
    }
}

}  // namespace qe::expr
