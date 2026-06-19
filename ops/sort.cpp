//  WP-7: Sort (ORDER BY) implementation. See ops/sort.h.
//
//  TWO INDEPENDENT SORT ALGORITHMS (D10), deliberately NOT collapsed (§3/D17):
//
//   * COMPARISON path (build_perm_comparison): std::stable_sort of a row-index
//     array under a left-to-right tuple comparator. Reads raw column values and
//     branches on null/direction. Handles ANY key type / mix.
//
//   * RADIX fast-path (build_perm_radix): for all-integer keys (I32/I64/TS). Each
//     row's key tuple is normalized into a fixed-width, memcmp-able BIG-ENDIAN
//     byte string (an order-preserving sign-flip per value; per-key NULL-indicator
//     byte for NULLS FIRST/LAST; per-key bytewise complement for DESC). A stable
//     LSD byte radix (least-significant byte first) then sorts the permutation.
//     This is a separate body from the comparator — a sign-bit / null-byte slip
//     surfaces as radix != comparison (the mutation self-test exploits exactly
//     that). The whole-row GATHER (sort_internal.gather_rows) is where Highway
//     earns its keep.
//
//  Both paths are STABLE on the child's row order (equal key tuples keep original
//  order), so radix-result == comparison-result on identical inputs — for any
//  input, not just totally-ordered ones.
//
//  PIPELINE-BREAKER: open() drains the child into a dense MaterializedColumns and
//  computes the permutation; next() emits sorted rows in dense <=kOutBatch batches
//  via the gather. The known 0-row compact_column abort is unreachable: the drain
//  never calls it (see sort_internal.cpp) and empty input emits zero rows.

#include "ops/sort.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>

#include "ops/sort_internal.h"

namespace qe {

namespace sd = qe::sort_detail;

struct Sort::State {
    sd::MaterializedColumns mat;
    std::vector<std::uint32_t> perm;  // sorted order: output row r == mat row perm[r]
};

Sort::Sort(std::unique_ptr<Operator> child, std::vector<SortKey> keys)
    : child_(std::move(child)), keys_(std::move(keys)) {
    assert(!keys_.empty() && "Sort needs >=1 key");
    child_schema_ = child_->output_schema();
}

Schema Sort::output_schema() const {
    // Sort reorders rows; it never changes the column set or types.
    return child_schema_;
}

namespace {

// ---- COMPARISON path -------------------------------------------------------

// "Does row a come strictly before row b?" under the ORDER BY keys, left to right.
// Independent of the radix encoding (different code path on purpose).
bool less_row(const sd::MaterializedColumns& mat, const std::vector<SortKey>& keys,
              std::uint32_t a, std::uint32_t b) {
    for (const SortKey& k : keys) {
        const std::size_t c = k.col;
        const bool av = mat.is_valid(c, a);
        const bool bv = mat.is_valid(c, b);
        if (!av || !bv) {
            if (!av && !bv) continue;  // both NULL: equal on this key
            // Exactly one NULL. Null placement is ABSOLUTE (independent of dir).
            const bool a_null = !av;
            return (k.nulls == NullOrder::First) ? a_null : !a_null;
        }
        int cmp;
        if (mat.types[c] == Type::F64) {
            const double fa = mat.f64_at(c, a);
            const double fb = mat.f64_at(c, b);
            cmp = (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
        } else if (mat.types[c] == Type::STR) {
            // WP-7b: order STR by VALUE (lexicographic bytes via the owned dict),
            // never by raw code. std::string_view ordering == DuckDB VARCHAR ASCII.
            cmp = mat.str_at(c, a).compare(mat.str_at(c, b));
        } else {
            const std::int64_t ia = mat.int_at(c, a);
            const std::int64_t ib = mat.int_at(c, b);
            cmp = (ia < ib) ? -1 : (ia > ib) ? 1 : 0;
        }
        if (cmp != 0) return (k.dir == SortDir::Asc) ? (cmp < 0) : (cmp > 0);
    }
    return false;  // fully equal: stable_sort keeps original order
}

std::vector<std::uint32_t> build_perm_comparison(
    const sd::MaterializedColumns& mat, const std::vector<SortKey>& keys) {
    std::vector<std::uint32_t> perm(mat.n);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](std::uint32_t a, std::uint32_t b) {
                         return less_row(mat, keys, a, b);
                     });
    return perm;
}

// ---- RADIX fast-path -------------------------------------------------------

// Number of value bytes used to encode key column `c` (TS/I64 -> 8, I32 -> 4).
std::size_t value_bytes(Type t) { return t == Type::I32 ? 4 : 8; }

// Order-preserving unsigned image of a signed integer value at (col,row): flip the
// sign bit so ascending unsigned == ascending signed. Returned right-aligned in a
// uint64; the caller emits the low `value_bytes` big-endian.
std::uint64_t encode_value(const sd::MaterializedColumns& mat, std::size_t col,
                           std::size_t row) {
    if (mat.types[col] == Type::I32) {
        const auto v = static_cast<std::int32_t>(mat.int_at(col, row));
        return static_cast<std::uint32_t>(v) ^ 0x80000000u;
    }
    const std::int64_t v = mat.int_at(col, row);
    return static_cast<std::uint64_t>(v) ^ 0x8000000000000000ull;
}

std::vector<std::uint32_t> build_perm_radix(const sd::MaterializedColumns& mat,
                                            const std::vector<SortKey>& keys) {
    const std::size_t n = mat.n;
    const std::size_t nk = keys.size();

    // Per-key field layout: [1 null-indicator byte][value bytes], key0 most
    // significant (lowest offset). Total key width W.
    std::vector<std::size_t> vbytes(nk), foff(nk);
    std::size_t W = 0;
    for (std::size_t j = 0; j < nk; ++j) {
        vbytes[j] = value_bytes(mat.types[keys[j].col]);
        foff[j] = W;
        W += 1 + vbytes[j];
    }

    std::vector<std::uint8_t> key(n * W, 0);
    for (std::size_t r = 0; r < n; ++r) {
        std::uint8_t* row = key.data() + r * W;
        for (std::size_t j = 0; j < nk; ++j) {
            const SortKey& k = keys[j];
            const std::size_t vb = vbytes[j];
            const std::size_t off = foff[j];
            const bool is_null = !mat.is_valid(k.col, r);

            // Null-indicator byte: position is set by NULLS FIRST/LAST only, NOT
            // by ASC/DESC (null placement is absolute). null-first => null byte 0.
            const std::uint8_t nb =
                (k.nulls == NullOrder::First) ? (is_null ? 0x00 : 0x01)
                                              : (is_null ? 0x01 : 0x00);
            row[off] = nb;

            const std::uint64_t u = is_null ? 0 : encode_value(mat, k.col, r);
            for (std::size_t b = 0; b < vb; ++b) {
                std::uint8_t byteval =
                    static_cast<std::uint8_t>(u >> (8 * (vb - 1 - b)));
                if (k.dir == SortDir::Desc)
                    byteval = static_cast<std::uint8_t>(~byteval);
                row[off + 1 + b] = byteval;
            }
        }
    }

    // Stable LSD byte radix: least-significant byte (highest offset) first, so the
    // most-significant byte (key0) decides last and dominates.
    std::vector<std::uint32_t> perm(n), tmp(n);
    std::iota(perm.begin(), perm.end(), 0u);
    if (W == 0) return perm;  // no key bytes (cannot happen: >=1 key) — be total
    for (std::size_t pos = W; pos-- > 0;) {
        std::size_t cnt[256] = {0};
        for (std::size_t i = 0; i < n; ++i) ++cnt[key[perm[i] * W + pos]];
        std::size_t off[256];
        std::size_t s = 0;
        for (int b = 0; b < 256; ++b) {
            off[b] = s;
            s += cnt[b];
        }
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint8_t bb = key[perm[i] * W + pos];
            tmp[off[bb]++] = perm[i];
        }
        perm.swap(tmp);
    }
    return perm;
}

}  // namespace

void Sort::open() {
    if (opened_) return;
    opened_ = true;
    state_ = std::make_shared<State>();
    State& st = *state_;

    st.mat = sd::materialize(*child_, child_schema_);

    bool use_radix;
    switch (path_) {
        case Path::kComparison:
            use_radix = false;
            break;
        case Path::kRadix:
            assert(sd::radix_eligible(keys_, child_schema_) &&
                   "Path::kRadix forced on non-integer keys");
            use_radix = true;
            break;
        case Path::kAuto:
        default:
            use_radix = sd::radix_eligible(keys_, child_schema_);
            break;
    }

    st.perm = use_radix ? build_perm_radix(st.mat, keys_)
                        : build_perm_comparison(st.mat, keys_);
    emit_cursor_ = 0;
}

std::optional<Batch> Sort::next() {
    assert(opened_ && "next() before open()");
    State& st = *state_;
    const std::size_t n = st.mat.num_rows();
    if (emit_cursor_ >= n) return std::nullopt;  // empty input => zero rows
    const std::size_t m = std::min(kOutBatch, n - emit_cursor_);

    current_ = sd::gather_rows(st.mat, st.perm.data(), emit_cursor_, m,
                               gather_path_ == GatherPath::kVector);
    emit_cursor_ += m;
    return current_.view();
}

void Sort::close() {
    state_.reset();
    opened_ = false;
}

}  // namespace qe
