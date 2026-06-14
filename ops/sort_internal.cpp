//  WP-7: implementation of the shared sort plumbing. See ops/sort_internal.h.
#include "ops/sort_internal.h"

#include <cstring>
#include <optional>

#include "core/selection.h"
#include "core/validity.h"
#include "simd/gather_kernels.h"

namespace qe::sort_detail {

std::int64_t MaterializedColumns::int_at(std::size_t col, std::size_t row) const {
    const std::byte* p = data[col].data();
    switch (types[col]) {
        case Type::I32:
            return reinterpret_cast<const std::int32_t*>(p)[row];
        case Type::I64:
        case Type::TS:
            return reinterpret_cast<const std::int64_t*>(p)[row];
        case Type::BOOL:
            return reinterpret_cast<const std::uint8_t*>(p)[row] ? 1 : 0;
        case Type::F64:
            // Not order-preserving as an int; callers use f64_at for F64.
            return 0;
    }
    return 0;
}

double MaterializedColumns::f64_at(std::size_t col, std::size_t row) const {
    return reinterpret_cast<const double*>(data[col].data())[row];
}

MaterializedColumns materialize(Operator& child, const Schema& schema) {
    MaterializedColumns mat;
    const std::size_t ncol = schema.fields.size();
    mat.types.reserve(ncol);
    mat.data.resize(ncol);
    mat.valid.resize(ncol);
    std::vector<std::size_t> width(ncol);
    for (std::size_t c = 0; c < ncol; ++c) {
        mat.types.push_back(schema.fields[c].second);
        width[c] = byte_width(schema.fields[c].second);
    }

    child.open();
    while (std::optional<Batch> in = child.next()) {
        const Batch& b = *in;
        const std::size_t rows = b.row_count;
        for (std::size_t k = 0; k < rows; ++k) {
            const std::uint32_t phys = sel_at(b.sel, k);
            for (std::size_t c = 0; c < ncol; ++c) {
                const Column& col = b.cols[c];
                const bool ok =
                    col.all_valid || validity::get_bit(col.validity, phys);
                const std::byte* src = col.data + phys * width[c];
                mat.data[c].insert(mat.data[c].end(), src, src + width[c]);
                mat.valid[c].push_back(ok ? 1 : 0);
            }
        }
        mat.n += rows;
    }
    child.close();
    return mat;
}

bool radix_eligible(const std::vector<SortKey>& keys, const Schema& schema) {
    for (const SortKey& k : keys) {
        const Type t = schema.fields[k.col].second;
        if (t != Type::I32 && t != Type::I64 && t != Type::TS) return false;
    }
    return true;
}

OwnedBatch gather_rows(const MaterializedColumns& mat, const std::uint32_t* perm,
                       std::size_t start, std::size_t m,
                       bool use_vector_gather) {
    OwnedBatch out;
    const std::uint32_t* idx = perm + start;
    for (std::size_t c = 0; c < mat.num_cols(); ++c) {
        const Type t = mat.types[c];
        OwnedColumn oc = OwnedColumn::make(t, m);
        std::byte* dst = oc.mutable_data();
        const std::byte* src = mat.col_data(c);

        switch (t) {
            case Type::I32: {
                auto* d = reinterpret_cast<std::uint32_t*>(dst);
                const auto* s = reinterpret_cast<const std::uint32_t*>(src);
                if (use_vector_gather)
                    simd::gather32_vec(s, idx, m, d);
                else
                    simd::gather32_scalar(s, idx, m, d);
                break;
            }
            case Type::I64:
            case Type::TS:
            case Type::F64: {
                auto* d = reinterpret_cast<std::uint64_t*>(dst);
                const auto* s = reinterpret_cast<const std::uint64_t*>(src);
                if (use_vector_gather)
                    simd::gather64_vec(s, idx, m, d);
                else
                    simd::gather64_scalar(s, idx, m, d);
                break;
            }
            case Type::BOOL: {
                // 1-byte lanes: scalar gather only (no vector twin exists; see
                // simd/gather_kernels.h). The flag is honored for both paths.
                auto* d = reinterpret_cast<std::uint8_t*>(dst);
                const auto* s = reinterpret_cast<const std::uint8_t*>(src);
                simd::gather8_scalar(s, idx, m, d);
                break;
            }
        }

        // Rebuild validity for the gathered rows (scalar; the bitmap is not the
        // SIMD story). Only allocate a bitmap if a null actually lands here.
        if (!mat.valid[c].empty()) {
            for (std::size_t r = 0; r < m; ++r)
                if (!mat.is_valid(c, idx[r])) oc.set_null(r);
        }
        out.add_column(std::move(oc));
    }
    return out;
}

}  // namespace qe::sort_detail
