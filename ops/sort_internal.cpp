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
        case Type::STR:
            // STR is ordered by VALUE via str_at (comparison path); never radix-
            // encoded (radix_eligible excludes it) and never read as an int.
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
    mat.dicts.assign(ncol, nullptr);  // WP-7b: per-STR-column owned canonical dict
    std::vector<std::size_t> width(ncol);
    for (std::size_t c = 0; c < ncol; ++c) {
        const Type t = schema.fields[c].second;
        mat.types.push_back(t);
        width[c] = byte_width(t);
        if (t == Type::STR) mat.dicts[c] = std::make_shared<StringDict>();
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
                if (mat.types[c] == Type::STR) {
                    // WP-7b: re-intern the string VALUE into the owned dict and
                    // store the canonical code (the source dict dies after drain).
                    std::int32_t canon = 0;
                    if (ok) {
                        const auto src_code =
                            reinterpret_cast<const std::int32_t*>(col.data)[phys];
                        canon = mat.dicts[c]->intern(col.dict->at(src_code));
                    }
                    const auto* cb = reinterpret_cast<const std::byte*>(&canon);
                    mat.data[c].insert(mat.data[c].end(), cb, cb + 4);
                } else {
                    const std::byte* src = col.data + phys * width[c];
                    mat.data[c].insert(mat.data[c].end(), src, src + width[c]);
                }
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
            case Type::I32:
            case Type::STR: {  // WP-7b: STR = int32 dict code; gather the codes
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

        // WP-7b: STR output codes index the materialized owned dict (alive for the
        // sort's lifetime); attach it so the gathered column resolves correctly.
        if (t == Type::STR) oc.set_dict(mat.col_dict(c));

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
