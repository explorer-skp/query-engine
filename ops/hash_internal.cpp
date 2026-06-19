//  WP-4: implementation of the shared normalization + per-batch key prep
//  (ops/hash_internal.h). Plain C++ (no Highway here); the only vectorized step
//  is delegated to the hash-combine twin in ops/hash_kernels.h.

#include "ops/hash_internal.h"

#include <bit>
#include <cmath>
#include <cstring>

#include "core/selection.h"
#include "core/validity.h"
#include "ops/hash_kernels.h"

namespace qe::ops::detail {

namespace {
// Canonical quiet NaN bit pattern (all NaNs collapse to this as a key).
constexpr std::uint64_t kCanonicalNaN = 0x7ff8'0000'0000'0000ull;

template <typename T>
T load_as(const std::byte* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}
}  // namespace

std::uint64_t normalize_value(Type t, const std::byte* p) {
    switch (t) {
        case Type::I32:
            // zero-extend the 32-bit pattern; low 32 bits recover the value.
            return static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(load_as<std::int32_t>(p)));
        case Type::I64:
        case Type::TS:
            return std::bit_cast<std::uint64_t>(load_as<std::int64_t>(p));
        case Type::BOOL:
            // BOOL is one byte (0/1) per value (core/types.h).
            return static_cast<std::uint64_t>(load_as<std::uint8_t>(p));
        case Type::F64: {
            const double d = load_as<double>(p);
            if (std::isnan(d)) return kCanonicalNaN;
            if (d == 0.0) return 0ull;  // collapses -0.0 and +0.0
            return std::bit_cast<std::uint64_t>(d);
        }
        case Type::STR:
            // WP-7b: the int32 dictionary CODE, zero-extended (same as I32). This
            // groups by code, which is correct ONLY when one dict spans the whole
            // key column (GROUP BY on a scanned STR column — codes are consistent
            // and dedup'd). Across DIFFERENT dicts (a JOIN's build vs probe side),
            // codes are NOT comparable; the join operator therefore canonicalizes
            // STR keys to a shared value-id space (Type::I32) BEFORE the table, so
            // STR never reaches here in a join. Hashing a join's raw STR codes is
            // exactly the planted "by code not by value" mutant.
            return static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(load_as<std::int32_t>(p)));
    }
    return 0ull;  // unreachable; all Types handled
}

void normalize_and_hash(const KeyColumns& keys, std::size_t n,
                        const std::vector<Type>& types, std::uint64_t seed,
                        HashPath path,
                        std::vector<std::vector<std::uint64_t>>& words,
                        std::vector<std::uint64_t>& null_mask,
                        std::vector<std::uint64_t>& hash) {
    const std::size_t ncols = keys.num_cols;
    words.resize(ncols);
    for (auto& w : words) w.resize(n);
    null_mask.assign(n, 0ull);
    hash.assign(n, seed);

    for (std::size_t j = 0; j < ncols; ++j) {
        const Column& col = keys.cols[j];
        const Type t = types[j];
        const std::size_t width = byte_width(t);
        std::uint64_t* wj = words[j].data();
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint32_t phys = sel_at(keys.sel, k);
            const bool valid =
                col.all_valid || validity::get_bit(col.validity, phys);
            if (!valid) {
                wj[k] = kNullWord;
                null_mask[k] |= (std::uint64_t{1} << j);
            } else {
                wj[k] = normalize_value(t, col.data + phys * width);
            }
        }
        // Fold this column into the running per-row hash via the chosen twin.
        if (path == HashPath::kVector) {
            hash_combine_vec(hash.data(), wj, n);
        } else {
            hash_combine_scalar(hash.data(), wj, n);
        }
    }
}

}  // namespace qe::ops::detail
