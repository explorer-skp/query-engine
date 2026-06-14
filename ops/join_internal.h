//  WP-6: INTERNAL hash-join helpers shared by the real HashJoin operator
//  (ops/join.cpp) AND the test-only planted-mutant operator (ops/join_mutants.cpp).
//  NOT a frozen contract (the analog of ops/agg_internal.h / ops/hash_internal.h).
//
//  Centralizing the NON-mutated pieces here — the build-side materialized store,
//  the per-key-column view assembly, and the two output-column emitters (which is
//  where the build-row GATHER, reused from simd/gather_kernels.h, actually
//  happens) — lets the mutant be a faithful "real join minus one step": it reuses
//  everything here UNCHANGED and copies only the pair-generation / output-chunk
//  loop with a single planted defect, so the differential isolates that defect.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "core/validity.h"
#include "ops/join.h"  // GatherPath
#include "simd/gather_kernels.h"

namespace qe::ops::detail {

// Sentinel build-row index meaning "no build row" — a LEFT-join unmatched probe
// row, whose build columns are emitted as NULL. Distinct from any real build row
// index (build rows are 0..nrows-1, and nrows <= UINT32_MAX-1 in practice).
inline constexpr std::uint32_t kNullBuildRow = UINT32_MAX;

// The build side, fully materialized at open() into dense, column-major raw bytes
// plus a per-row validity flag per column (1 == valid). Row r (0..nrows-1) is one
// drained build-input row; the multiplicity map (group_id -> [row indices]) lives
// in the operator and indexes into this store. Stored densely so the gather
// kernels can read it by row index.
struct BuildStore {
    std::vector<Type> types;
    std::vector<std::vector<std::byte>> data;      // [col] nrows*byte_width(type)
    std::vector<std::vector<std::uint8_t>> valid;  // [col] nrows (1 == valid)
    std::size_t nrows = 0;

    void init(const std::vector<Type>& t) {
        types = t;
        data.assign(t.size(), {});
        valid.assign(t.size(), {});
        nrows = 0;
    }

    // Append physical row `p` of every column of batch `b` (cols selected by
    // `cols`, i.e. the build child's full column list) to the store. Returns the
    // new build-row index.
    std::uint32_t append_row(const Batch& b, std::size_t p) {
        for (std::size_t c = 0; c < types.size(); ++c) {
            const Column& col = b.cols[c];
            const std::size_t w = byte_width(types[c]);
            const std::byte* src = col.data + static_cast<std::size_t>(p) * w;
            data[c].insert(data[c].end(), src, src + w);
            const bool v = col.all_valid || validity::get_bit(col.validity, p);
            valid[c].push_back(v ? 1u : 0u);
        }
        return static_cast<std::uint32_t>(nrows++);
    }

    const std::byte* col_data(std::size_t c) const { return data[c].data(); }
};

// Gather the DATA words of one column by index `idx[0..m)` into `out` (length m),
// honoring the lane width of `t`. I32 -> gather32, I64/F64/TS -> gather64,
// BOOL -> the scalar byte gather (no vector twin). `out`/`src` never alias (out is
// a fresh OwnedColumn buffer), so the §12 aliasing hazard cannot arise.
inline void gather_data(OwnedColumn& out, Type t, const std::byte* src,
                        const std::uint32_t* idx, std::size_t m, GatherPath gp) {
    std::byte* dst = out.mutable_data();
    switch (t) {
        case Type::I32: {
            auto* d = reinterpret_cast<std::uint32_t*>(dst);
            const auto* s = reinterpret_cast<const std::uint32_t*>(src);
            if (gp == GatherPath::kVector)
                simd::gather32_vec(s, idx, m, d);
            else
                simd::gather32_scalar(s, idx, m, d);
            return;
        }
        case Type::I64:
        case Type::F64:
        case Type::TS: {
            auto* d = reinterpret_cast<std::uint64_t*>(dst);
            const auto* s = reinterpret_cast<const std::uint64_t*>(src);
            if (gp == GatherPath::kVector)
                simd::gather64_vec(s, idx, m, d);
            else
                simd::gather64_scalar(s, idx, m, d);
            return;
        }
        case Type::BOOL: {
            auto* d = reinterpret_cast<std::uint8_t*>(dst);
            const auto* s = reinterpret_cast<const std::uint8_t*>(src);
            simd::gather8_scalar(s, idx, m, d);  // no vector twin (documented)
            return;
        }
    }
}

// Emit one PROBE column for a chunk of m output rows: out[r] = src[phys_idx[r]],
// with validity carried from the source column. `phys_idx` are PHYSICAL probe row
// indices (the caller already resolved any probe selection vector).
inline void emit_probe_column(OwnedColumn& out, const Column& src,
                              const std::uint32_t* phys_idx, std::size_t m,
                              GatherPath gp) {
    gather_data(out, src.type, src.data, phys_idx, m, gp);
    if (!src.all_valid) {
        for (std::size_t r = 0; r < m; ++r)
            if (!validity::get_bit(src.validity, phys_idx[r])) out.set_null(r);
    }
}

// Emit one BUILD column (store column `c`) for a chunk of m output rows:
// out[r] = store.col(c)[build_idx[r]], or NULL where build_idx[r]==kNullBuildRow
// (a LEFT-join unmatched probe row). Validity is carried from the store. Uses a
// scratch index array so a sentinel never drives an out-of-range gather read.
inline void emit_build_column(OwnedColumn& out, const BuildStore& store,
                              std::size_t c, const std::uint32_t* build_idx,
                              std::size_t m, GatherPath gp,
                              std::vector<std::uint32_t>& scratch_idx) {
    if (store.nrows == 0) {
        // No build rows at all (LEFT join, every probe row unmatched): the entire
        // column is NULL. Skip the gather (gathering from an empty source would be
        // an out-of-range read).
        for (std::size_t r = 0; r < m; ++r) out.set_null(r);
        return;
    }
    scratch_idx.resize(m);
    for (std::size_t r = 0; r < m; ++r) {
        const std::uint32_t b = build_idx[r];
        scratch_idx[r] = (b == kNullBuildRow) ? 0u : b;  // sentinel -> safe row 0
    }
    gather_data(out, store.types[c], store.col_data(c), scratch_idx.data(), m, gp);
    const std::vector<std::uint8_t>& v = store.valid[c];
    for (std::size_t r = 0; r < m; ++r) {
        const std::uint32_t b = build_idx[r];
        if (b == kNullBuildRow || !v[b]) out.set_null(r);
    }
}

}  // namespace qe::ops::detail
