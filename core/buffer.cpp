//  WP-1: Buffer implementation. See core/buffer.h. Memory comes from the SIMD
//  layer's aligned allocator (simd/aligned_alloc.h), so core/ pulls in no
//  third-party SIMD headers — the from-scratch grep stays clean.

#include "core/buffer.h"

#include <cassert>
#include <cstdint>

#include "simd/aligned_alloc.h"

namespace qe {

Buffer::Buffer(std::size_t nbytes) {
    if (nbytes == 0) return;
    data_ = static_cast<std::byte*>(simd::aligned_alloc_bytes(nbytes));
    size_ = nbytes;
    // The runtime-derived alignment contract must actually hold.
    assert(reinterpret_cast<std::uintptr_t>(data_) %
                   simd::required_alignment_bytes() ==
               0 &&
           "Buffer allocation did not meet the runtime SIMD alignment");
}

Buffer::~Buffer() { simd::aligned_free_bytes(data_); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        simd::aligned_free_bytes(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

std::size_t Buffer::alignment() const noexcept {
    return size_ == 0 ? 0 : simd::required_alignment_bytes();
}

}  // namespace qe
