//  WP-7: INTERNAL helpers shared by the real Sort operator (ops/sort.cpp) and the
//  TEST-ONLY planted mutant (ops/sort_mutants.cpp). NOT a frozen contract — the
//  analog of ops/agg_internal.h. These are the plumbing pieces that are NOT the
//  sort algorithm itself (so reusing them keeps the mutant a faithful copy that
//  differs from the real operator by exactly one step):
//   * MaterializedColumns — the dense drain of a child's output.
//   * radix_eligible      — key-type classification for the radix fast-path.
//   * gather_rows         — whole-row permutation gather into an output batch.
//
//  The two SORT ALGORITHMS themselves (radix encode + LSD passes; the comparison
//  tuple comparator) live in sort.cpp / sort_mutants.cpp as INDEPENDENT code, per
//  §3/D17 — they are deliberately NOT shared here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/operator.h"
#include "ops/sort.h"

namespace qe::sort_detail {

// A dense, fully-materialized copy of a drained operator's output: one contiguous
// data buffer (n * byte_width(type)) plus one validity flag per row, per column.
// Built by draining the child fully and honoring each batch's optional selection
// vector and the validity precedence. It deliberately does NOT route through
// core/owned_batch.h compact_column (whose n==0 path aborts — reviewer-owned),
// so empty input is represented cleanly as num_rows()==0.
struct MaterializedColumns {
    std::vector<Type> types;
    std::vector<std::vector<std::byte>> data;       // [col] -> n*width bytes
    std::vector<std::vector<std::uint8_t>> valid;   // [col] -> n flags (1=valid)
    std::size_t n = 0;

    std::size_t num_rows() const { return n; }
    std::size_t num_cols() const { return types.size(); }

    bool is_valid(std::size_t col, std::size_t row) const {
        return valid[col][row] != 0;
    }

    // Read an integer-family value (I32/I64/TS/BOOL) at (col,row) as int64. The
    // mapping is order-preserving for the source type. Precondition: the column is
    // integer-family (the comparison path's float keys use f64_at instead).
    std::int64_t int_at(std::size_t col, std::size_t row) const;
    double f64_at(std::size_t col, std::size_t row) const;

    const std::byte* col_data(std::size_t col) const { return data[col].data(); }
};

// Drain `child` (already constructed; this calls open()/next()/close()) into a
// dense MaterializedColumns matching `schema`.
MaterializedColumns materialize(Operator& child, const Schema& schema);

// True iff EVERY key column is fixed-width integer/timestamp (I32/I64/TS) — the
// radix fast-path's eligibility test.
bool radix_eligible(const std::vector<SortKey>& keys, const Schema& schema);

// Gather output rows perm[start], perm[start+1], ..., perm[start+m-1] of `mat`
// into a fresh dense OwnedBatch (one column per input column). The DATA bytes move
// through the Highway gather kernels (use_vector_gather) or their scalar twins;
// validity is rebuilt with a scalar loop (the bitmap is not the SIMD story).
// Out-of-place by construction (the §12 selection-vector aliasing hazard cannot
// arise: source and destination are distinct buffers).
OwnedBatch gather_rows(const MaterializedColumns& mat, const std::uint32_t* perm,
                       std::size_t start, std::size_t m, bool use_vector_gather);

}  // namespace qe::sort_detail
