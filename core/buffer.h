//  WP-1: the owning, aligned, contiguous byte allocation that backs every
//  Column/Batch view.
//
//  OWNERSHIP MODEL (frozen at WP-1):
//   * Buffer OWNS its bytes. It is the only owning primitive in the format; it is
//     move-only (unique ownership, no reference counting) and frees on destruct.
//   * Column / Batch / SelectionVector are NON-OWNING VIEWS into Buffers (see
//     core/column.h). A view never outlives the Buffer(s) it points into.
//   * OwnedColumn / OwnedBatch (core/owned_batch.h) bundle the Buffer(s) for one
//     column / batch and hand out the matching views; they are the owning layer
//     operators allocate from.
//
//  ALIGNMENT: a Buffer's bytes are aligned to at least
//  simd::required_alignment_bytes() (runtime-derived from Highway's vector width
//  — never a hardcoded 16/64). data() is therefore always safe for aligned SIMD
//  loads/stores of any lane type on the dispatched target.
#pragma once

#include <cstddef>
#include <utility>

namespace qe {

class Buffer {
   public:
    // Empty buffer: data() == nullptr, size() == 0.
    Buffer() noexcept = default;

    // Allocate `nbytes` aligned to simd::required_alignment_bytes(). Contents are
    // uninitialized. nbytes == 0 yields an empty buffer (no allocation).
    explicit Buffer(std::size_t nbytes);

    ~Buffer();

    // Move-only: ownership is unique.
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::byte* data() noexcept { return data_; }
    const std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // The alignment (bytes) data() is guaranteed to satisfy. 0 for an empty
    // buffer. Runtime-derived; see simd::required_alignment_bytes().
    std::size_t alignment() const noexcept;

    void swap(Buffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

   private:
    std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace qe
