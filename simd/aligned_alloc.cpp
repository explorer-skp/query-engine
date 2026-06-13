//  WP-1: aligned allocation + runtime vector-width detection. See
//  simd/aligned_alloc.h and simd/kernel_convention.h. This translation unit is
//  one of the few places Highway is included; it links the `hwy` target and is
//  built with -Wno-error (it pulls in third-party Highway headers we do not
//  patch). It contains NO SIMD kernels — only Highway's runtime-width query and
//  its aligned allocator, wrapped so core/ stays Highway-free.

#include "simd/aligned_alloc.h"

#include "simd/kernel_convention.h"

#include "hwy/aligned_allocator.h"  // hwy::AllocateAlignedBytes / FreeAlignedBytes
#include "hwy/per_target.h"         // hwy::VectorBytes (runtime-detected width)

namespace qe::simd {

std::size_t runtime_vector_bytes() {
    // Highway reports the widest vector (in bytes) for the target it dispatched
    // to on this CPU. Always > 0.
    const std::size_t v = hwy::VectorBytes();
    return v == 0 ? 1 : v;
}

std::size_t required_alignment_bytes() {
    // Round the runtime vector width up to the next power of two. Derived, not
    // literal: NEON -> 16, AVX-512 -> 64, etc.
    std::size_t v = runtime_vector_bytes();
    std::size_t a = 1;
    while (a < v) a <<= 1;
    return a;
}

void* aligned_alloc_bytes(std::size_t nbytes) {
    if (nbytes == 0) return nullptr;
    // Highway's allocator aligns to at least HWY_ALIGNMENT (>= any SIMD vector on
    // any supported target) and adds cache-line padding, so the pointer it
    // returns always satisfies required_alignment_bytes(). The two-arg form with
    // null alloc/free uses Highway's default backing allocator.
    return hwy::AllocateAlignedBytes(nbytes, nullptr, nullptr);
}

void aligned_free_bytes(void* p) noexcept {
    if (p == nullptr) return;
    hwy::FreeAlignedBytes(p, nullptr, nullptr);
}

}  // namespace qe::simd
