//  WP-2: SCALAR TWIN for the three-valued logic kernels (see expr/kernels.h).
//
//  Operates on TRI-STATE bytes: 0 = FALSE, 1 = NULL, 2 = TRUE. This twin is
//  written as EXPLICIT truth-table branches — deliberately NOT as the min/max/
//  (2-x) arithmetic the vector path (expr/logic_kernels.cpp) uses — so that
//  `scalar == vector` compares two independent encodings of Kleene logic, not
//  one. Inputs are guaranteed in {0,1,2} by eval.

#include <cstddef>
#include <cstdint>

#include "expr/kernels.h"

namespace qe::expr {

void logic_and_scalar(const std::uint8_t* a, const std::uint8_t* b,
                      std::uint8_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        // FALSE dominates (F∧N=F); else NULL dominates; else both TRUE.
        if (a[i] == 0 || b[i] == 0) {
            out[i] = 0;
        } else if (a[i] == 1 || b[i] == 1) {
            out[i] = 1;
        } else {
            out[i] = 2;
        }
    }
}

void logic_or_scalar(const std::uint8_t* a, const std::uint8_t* b,
                     std::uint8_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        // TRUE dominates (T∨N=T); else NULL dominates; else both FALSE.
        if (a[i] == 2 || b[i] == 2) {
            out[i] = 2;
        } else if (a[i] == 1 || b[i] == 1) {
            out[i] = 1;
        } else {
            out[i] = 0;
        }
    }
}

void logic_not_scalar(const std::uint8_t* a, std::uint8_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        // ¬T=F, ¬F=T, ¬N=N.
        if (a[i] == 0) {
            out[i] = 2;
        } else if (a[i] == 2) {
            out[i] = 0;
        } else {
            out[i] = 1;
        }
    }
}

}  // namespace qe::expr
