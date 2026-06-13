//  WP-1: aligned allocation for the columnar format, behind the SIMD layer.
//
//  Highway is the project's portable-SIMD dependency and appears ONLY under
//  simd/. core/buffer.cpp gets its aligned memory through these wrappers rather
//  than including Highway directly, which keeps core/ free of third-party SIMD
//  headers and keeps the from-scratch grep honest.
//
//  Alignment is RUNTIME-DERIVED, never a literal: required_alignment_bytes() is
//  computed from simd::runtime_vector_bytes() (Highway's detected vector width),
//  so a wider ISA (AVX-512) widens the guarantee with no source change.
#pragma once

#include <cstddef>

namespace qe::simd {

// The alignment, in bytes, that aligned_alloc_bytes() guarantees: a power of two
// >= the runtime SIMD vector width (simd::runtime_vector_bytes()). Derived at
// runtime; the underlying allocator may over-align beyond this.
std::size_t required_alignment_bytes();

// Allocate `nbytes` aligned to at least required_alignment_bytes(). Returns
// nullptr iff nbytes == 0. The result must be released with aligned_free_bytes().
void* aligned_alloc_bytes(std::size_t nbytes);

// Release memory from aligned_alloc_bytes(). nullptr is a no-op.
void aligned_free_bytes(void* p) noexcept;

}  // namespace qe::simd
