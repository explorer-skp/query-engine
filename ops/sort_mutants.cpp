//  WP-7: implementation of the TEST-ONLY mutant Sort (ops/sort_mutants.h). A
//  faithful copy of ops/sort.cpp's drain / permutation / emit loops, reusing the
//  SAME ops/sort_internal.h helpers, with exactly one planted defect per
//  SortMutation. The `// BUG:` lines mark each deviation from the real operator.

#include "ops/sort_mutants.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <utility>

#include "ops/sort_internal.h"

namespace qe::mutant {

namespace sd = qe::sort_detail;

struct Sort::State {
    sd::MaterializedColumns mat;
    std::vector<std::uint32_t> perm;
};

Sort::Sort(std::unique_ptr<Operator> child, std::vector<SortKey> keys,
           SortMutation mut)
    : child_(std::move(child)), keys_(std::move(keys)), mut_(mut) {
    assert(!keys_.empty());
    child_schema_ = child_->output_schema();
}

Schema Sort::output_schema() const { return child_schema_; }

namespace {

// Faithful copy of the real comparison comparator, with the kDescSortsAsc /
// kNullsFlipped / kUnstableTiebreak defects threaded in.
bool less_row_mut(const sd::MaterializedColumns& mat,
                  const std::vector<SortKey>& keys, std::uint32_t a,
                  std::uint32_t b, SortMutation mut) {
    for (const SortKey& k : keys) {
        const std::size_t c = k.col;
        const bool av = mat.is_valid(c, a);
        const bool bv = mat.is_valid(c, b);
        if (!av || !bv) {
            if (!av && !bv) continue;
            const bool a_null = !av;
            NullOrder nulls = k.nulls;
            if (mut == SortMutation::kNullsFlipped) {
                // BUG: flip NULLS FIRST/LAST.
                nulls = (nulls == NullOrder::First) ? NullOrder::Last
                                                    : NullOrder::First;
            }
            return (nulls == NullOrder::First) ? a_null : !a_null;
        }
        int cmp;
        if (mat.types[c] == Type::F64) {
            const double fa = mat.f64_at(c, a);
            const double fb = mat.f64_at(c, b);
            cmp = (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
        } else {
            const std::int64_t ia = mat.int_at(c, a);
            const std::int64_t ib = mat.int_at(c, b);
            cmp = (ia < ib) ? -1 : (ia > ib) ? 1 : 0;
        }
        if (cmp != 0) {
            SortDir dir = k.dir;
            if (mut == SortMutation::kDescSortsAsc) {
                // BUG: ignore DESC; always sort ascending.
                dir = SortDir::Asc;
            }
            return (dir == SortDir::Asc) ? (cmp < 0) : (cmp > 0);
        }
    }
    // Fully equal on all keys.
    if (mut == SortMutation::kUnstableTiebreak) {
        // BUG: break ties by REVERSED original index instead of keeping order.
        return a > b;
    }
    return false;
}

std::vector<std::uint32_t> build_perm_comparison_mut(
    const sd::MaterializedColumns& mat, const std::vector<SortKey>& keys,
    SortMutation mut) {
    std::vector<std::uint32_t> perm(mat.n);
    std::iota(perm.begin(), perm.end(), 0u);
    // The real operator uses std::stable_sort; with the kUnstableTiebreak defect
    // the comparator is a TOTAL order (reversed-index tiebreak), so plain sort
    // realizes the reversed tie order deterministically.
    if (mut == SortMutation::kUnstableTiebreak) {
        std::sort(perm.begin(), perm.end(),
                  [&](std::uint32_t a, std::uint32_t b) {
                      return less_row_mut(mat, keys, a, b, mut);
                  });
    } else {
        std::stable_sort(perm.begin(), perm.end(),
                         [&](std::uint32_t a, std::uint32_t b) {
                             return less_row_mut(mat, keys, a, b, mut);
                         });
    }
    return perm;
}

std::size_t value_bytes(Type t) { return t == Type::I32 ? 4 : 8; }

// Copy of the radix encode with the kRadixSignBug defect.
std::uint64_t encode_value_mut(const sd::MaterializedColumns& mat,
                               std::size_t col, std::size_t row,
                               SortMutation mut) {
    if (mut == SortMutation::kRadixSignBug) {
        // BUG: omit the sign-bit flip. Negative integers (high bit set) then
        // encode larger than non-negatives and sort AFTER them.
        if (mat.types[col] == Type::I32)
            return static_cast<std::uint32_t>(
                static_cast<std::int32_t>(mat.int_at(col, row)));
        return static_cast<std::uint64_t>(mat.int_at(col, row));
    }
    if (mat.types[col] == Type::I32) {
        const auto v = static_cast<std::int32_t>(mat.int_at(col, row));
        return static_cast<std::uint32_t>(v) ^ 0x80000000u;
    }
    const std::int64_t v = mat.int_at(col, row);
    return static_cast<std::uint64_t>(v) ^ 0x8000000000000000ull;
}

std::vector<std::uint32_t> build_perm_radix_mut(
    const sd::MaterializedColumns& mat, const std::vector<SortKey>& keys,
    SortMutation mut) {
    const std::size_t n = mat.n;
    const std::size_t nk = keys.size();
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
            const std::uint8_t nb =
                (k.nulls == NullOrder::First) ? (is_null ? 0x00 : 0x01)
                                              : (is_null ? 0x01 : 0x00);
            row[off] = nb;
            const std::uint64_t u =
                is_null ? 0 : encode_value_mut(mat, k.col, r, mut);
            for (std::size_t b = 0; b < vb; ++b) {
                std::uint8_t byteval =
                    static_cast<std::uint8_t>(u >> (8 * (vb - 1 - b)));
                if (k.dir == SortDir::Desc)
                    byteval = static_cast<std::uint8_t>(~byteval);
                row[off + 1 + b] = byteval;
            }
        }
    }
    std::vector<std::uint32_t> perm(n), tmp(n);
    std::iota(perm.begin(), perm.end(), 0u);
    if (W == 0) return perm;
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

    // kRadixSignBug lives in the radix path; the rest in the comparison path. (The
    // real operator's auto-selection picks radix for all-integer keys; the test
    // chooses key types to route each mutant to its planted path.)
    if (mut_ == SortMutation::kRadixSignBug) {
        assert(sd::radix_eligible(keys_, child_schema_) &&
               "kRadixSignBug needs integer keys");
        st.perm = build_perm_radix_mut(st.mat, keys_, mut_);
    } else {
        st.perm = build_perm_comparison_mut(st.mat, keys_, mut_);
    }
    emit_cursor_ = 0;
}

std::optional<Batch> Sort::next() {
    assert(opened_);
    State& st = *state_;
    std::size_t n = st.mat.num_rows();
    if (mut_ == SortMutation::kEmitTailOffByOne && n > 0) {
        // BUG: drop the last sorted row (short final batch -> row-count mismatch).
        n = n - 1;
    }
    if (emit_cursor_ >= n) return std::nullopt;
    const std::size_t m = std::min(kOutBatch, n - emit_cursor_);
    current_ = sd::gather_rows(st.mat, st.perm.data(), emit_cursor_, m,
                               /*use_vector_gather=*/true);
    emit_cursor_ += m;
    return current_.view();
}

void Sort::close() {
    state_.reset();
    opened_ = false;
}

}  // namespace qe::mutant
