//  WP-14: SCALAR TWIN for the zig-zag-decode kernel (see tsx/compress_kernels.h).
//  Deliberately independent, plain-C++ code — NOT the vector body with the SIMD
//  switched off (collapsing them would make scalar==vector prove nothing; rule 3 /
//  D17). Keeps the full -Wall -Wextra -Werror gate (no third-party headers).

#include "tsx/compress_kernels.h"

namespace qe::tsx {

void unzigzag_scalar(const std::uint64_t* zz, std::size_t n, std::uint64_t* out) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t v = zz[i];
        out[i] = (v >> 1) ^ (0ull - (v & 1ull));
    }
}

}  // namespace qe::tsx
