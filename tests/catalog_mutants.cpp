//  WP-9: implementations of the three new catalog mutants. See catalog_mutants.h
//  for the hazard each represents. Each is the DELIBERATELY-WRONG twin of a real
//  engine step; the real step lives in the engine libraries and is what the
//  catalog check() compares against.

#include "tests/catalog_mutants.h"

#include <cmath>
#include <cstdint>

namespace qe::catalog::mutant {

void cast_f64_to_i64_plus_half(const double* in, std::int64_t* out,
                               std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        // The bug: bias by +/-0.5 then truncate toward zero. Correct rounding
        // would truncate the (already-integral) value unchanged.
        const double biased = x + (x >= 0.0 ? 0.5 : -0.5);
        out[i] = static_cast<std::int64_t>(std::trunc(biased));
    }
}

std::int64_t sum_i64_wrapping_i32(const std::int64_t* in, std::size_t n) {
    // The bug: a 32-bit accumulator. Unsigned so the wraparound is well-defined
    // (no signed-overflow UB); the divergence from the I64 truth is the point.
    std::uint32_t acc = 0;
    for (std::size_t i = 0; i < n; ++i)
        acc += static_cast<std::uint32_t>(static_cast<std::int32_t>(in[i]));
    return static_cast<std::int64_t>(static_cast<std::int32_t>(acc));
}

void gather_in_place_aliased(std::int64_t* buf, const std::uint32_t* idx,
                             std::size_t n) {
    // The bug: in-place (out == in). A later index that points at an already-
    // overwritten slot reads corrupted data — the §12 aliasing hazard the real
    // out-of-place compact_column is designed to avoid.
    for (std::size_t k = 0; k < n; ++k) buf[k] = buf[idx[k]];
}

}  // namespace qe::catalog::mutant
