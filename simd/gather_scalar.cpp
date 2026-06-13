//  WP-1: SCALAR TWINS for the selection-vector gather kernels (see
//  simd/gather_kernels.h). Plain, Highway-free reference loops — independent of
//  the Highway GatherIndex path in simd/gather_kernels.cpp.

#include "simd/gather_kernels.h"

namespace qe::simd {

void gather32_scalar(const std::uint32_t* src, const std::uint32_t* idx,
                     std::size_t n, std::uint32_t* out) {
    if (idx == nullptr) {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[k];
    } else {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[idx[k]];
    }
}

void gather64_scalar(const std::uint64_t* src, const std::uint32_t* idx,
                     std::size_t n, std::uint64_t* out) {
    if (idx == nullptr) {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[k];
    } else {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[idx[k]];
    }
}

void gather8_scalar(const std::uint8_t* src, const std::uint32_t* idx,
                    std::size_t n, std::uint8_t* out) {
    if (idx == nullptr) {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[k];
    } else {
        for (std::size_t k = 0; k < n; ++k) out[k] = src[idx[k]];
    }
}

}  // namespace qe::simd
