//  WP-14 mutation self-test support. See tsx/compress_mutants.h. A faithful copy of
//  tsx/compress.cpp's streaming decode over the SAME EncodedColumn format and the
//  SAME zig-zag kernel, differing by exactly one planted decode defect.

#include "tsx/compress_mutants.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>

#include "core/types.h"
#include "core/validity.h"
#include "tsx/compress_kernels.h"

namespace qe::mutant {

using qe::tsx::CompressedTable;
using qe::tsx::EncodedColumn;

namespace {

// --- bit / varint readers (copy of the real decoder's primitives) -----------

struct BitReader {
    const std::byte* data;
    std::size_t bitpos;
    unsigned get_bit() {
        const unsigned byte =
            static_cast<unsigned>(static_cast<unsigned char>(data[bitpos >> 3]));
        const unsigned bit = (byte >> (7u - (bitpos & 7u))) & 1u;
        ++bitpos;
        return bit;
    }
    std::uint64_t get_bits(int n) {
        std::uint64_t v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | get_bit();
        return v;
    }
};

std::uint64_t get_varint(const std::byte* data, std::size_t& pos) {
    std::uint64_t v = 0;
    int shift = 0;
    while (true) {
        const unsigned char b = static_cast<unsigned char>(data[pos++]);
        v |= static_cast<std::uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0) break;
        shift += 7;
    }
    return v;
}

bool valid_bit(const EncodedColumn& e, std::size_t row) {
    if (e.all_valid) return true;
    return validity::get_bit(
        reinterpret_cast<const std::uint64_t*>(e.validity.data()), row);
}

struct ColCursor {
    std::size_t byte_pos = 0;
    std::size_t bit_pos = 0;
    std::uint64_t prev = 0;
    std::uint64_t prev_delta = 0;
    std::uint64_t g_prev = 0;
    unsigned g_lead = 0;
    unsigned g_trail = 0;
    unsigned g_mean = 0;
    bool g_started = false;
};

}  // namespace

struct CompressedScan::State {
    std::vector<ColCursor> cur;
    std::vector<std::uint64_t> zz;
    std::vector<std::uint64_t> dec;
    std::vector<std::uint32_t> slots;
};

CompressedScan::CompressedScan(const CompressedTable& ct, CompressMutation mut,
                              std::size_t batch_size)
    : ct_(ct), mut_(mut), batch_size_(batch_size) {
    if (batch_size_ == 0) batch_size_ = Scan::kDefaultBatchSize;
}

Schema CompressedScan::output_schema() const { return ct_.schema(); }

void CompressedScan::open() {
    opened_ = true;
    cursor_ = 0;
    state_ = std::make_shared<State>();
    state_->cur.assign(ct_.num_columns(), ColCursor{});
}

void CompressedScan::close() {
    cursor_ = ct_.num_rows();
    state_.reset();
    opened_ = false;
}

namespace {

void decode_int_window(const EncodedColumn& e, ColCursor& cc, OwnedColumn& oc,
                       std::size_t start, std::size_t n, CompressMutation mut,
                       std::vector<std::uint64_t>& zz,
                       std::vector<std::uint64_t>& dec,
                       std::vector<std::uint32_t>& slots) {
    const std::byte* vbytes = e.values.data();
    slots.clear();
    zz.clear();
    for (std::size_t k = 0; k < n; ++k) {
        if (!valid_bit(e, start + k)) {
            oc.set_null(k);
            continue;
        }
        slots.push_back(static_cast<std::uint32_t>(k));
        zz.push_back(get_varint(vbytes, cc.byte_pos));
    }
    const std::size_t m = zz.size();
    dec.resize(m);
    if (m) qe::tsx::unzigzag_scalar(zz.data(), m, dec.data());
    auto* out = reinterpret_cast<std::byte*>(oc.mutable_data());
    const bool dod = (e.type == Type::I64 || e.type == Type::TS);
    for (std::size_t j = 0; j < m; ++j) {
        std::uint64_t bits;
        if (dod) {
            // DEFECT kDropSecondDerivative: ignore the running prev_delta term, so
            // each dod is mis-treated as a plain first-order delta and the series
            // drifts. (The real decoder accumulates delta = prev_delta + dod.)
            if (mut == CompressMutation::kDropSecondDerivative) {
                const std::uint64_t val = cc.prev + dec[j];
                cc.prev = val;
                bits = val;
            } else {
                const std::uint64_t delta = cc.prev_delta + dec[j];
                const std::uint64_t val = cc.prev + delta;
                cc.prev_delta = delta;
                cc.prev = val;
                bits = val;
            }
        } else {
            bits = dec[j];
        }
        const std::uint32_t slot = slots[j];
        if (e.type == Type::I32)
            reinterpret_cast<std::int32_t*>(out)[slot] =
                static_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
        else
            std::memcpy(out + static_cast<std::size_t>(slot) * 8, &bits, 8);
    }
}

void decode_f64_window(const EncodedColumn& e, ColCursor& cc, OwnedColumn& oc,
                       std::size_t start, std::size_t n, CompressMutation mut) {
    BitReader br{e.values.data(), cc.bit_pos};
    auto* out = reinterpret_cast<double*>(oc.mutable_data());
    for (std::size_t k = 0; k < n; ++k) {
        if (!valid_bit(e, start + k)) {
            oc.set_null(k);
            continue;
        }
        std::uint64_t value;
        if (!cc.g_started) {
            value = br.get_bits(64);
            cc.g_started = true;
        } else if (br.get_bit() == 0) {
            value = cc.g_prev;
        } else {
            std::uint64_t x;
            if (br.get_bit() == 0) {
                const std::uint64_t meaningful =
                    br.get_bits(static_cast<int>(cc.g_mean));
                x = meaningful << cc.g_trail;
            } else {
                const unsigned L = static_cast<unsigned>(br.get_bits(5));
                const unsigned m6 = static_cast<unsigned>(br.get_bits(6));
                const unsigned mean = (m6 == 0u) ? 64u : m6;
                const std::uint64_t meaningful =
                    br.get_bits(static_cast<int>(mean));
                // DEFECT kGorillaLeadingZerosOff: re-derive the trailing-zero shift
                // from (L-1) instead of L, shifting the meaningful block up by one
                // bit whenever the block had ≥1 leading zero. (Shift stays <= 63, so
                // no UB; the reconstructed double is simply wrong by bits.)
                const unsigned Leff =
                    (mut == CompressMutation::kGorillaLeadingZerosOff && L > 0u)
                        ? (L - 1u)
                        : L;
                const unsigned trail = 64u - Leff - mean;
                x = meaningful << trail;
                cc.g_lead = L;
                cc.g_trail = 64u - L - mean;  // window state tracks the TRUE trail
                cc.g_mean = mean;
            }
            value = cc.g_prev ^ x;
        }
        cc.g_prev = value;
        std::memcpy(out + k, &value, 8);
    }
    cc.bit_pos = br.bitpos;
}

void decode_bool_window(const EncodedColumn& e, ColCursor& cc, OwnedColumn& oc,
                        std::size_t start, std::size_t n) {
    BitReader br{e.values.data(), cc.bit_pos};
    auto* out = reinterpret_cast<std::uint8_t*>(oc.mutable_data());
    for (std::size_t k = 0; k < n; ++k) {
        if (!valid_bit(e, start + k)) {
            oc.set_null(k);
            continue;
        }
        out[k] = static_cast<std::uint8_t>(br.get_bit());
    }
    cc.bit_pos = br.bitpos;
}

}  // namespace

std::optional<Batch> CompressedScan::next() {
    assert(opened_ && "next() before open()");
    const std::size_t total = ct_.num_rows();
    if (cursor_ >= total) return std::nullopt;
    const std::size_t start = cursor_;
    const std::size_t n = std::min(batch_size_, total - start);

    OwnedBatch out;
    for (std::size_t c = 0; c < ct_.num_columns(); ++c) {
        const EncodedColumn& e = ct_.encoded(c);
        OwnedColumn oc = OwnedColumn::make(e.type, n);
        std::memset(oc.mutable_data(), 0, n * byte_width(e.type));
        ColCursor& cc = state_->cur[c];
        switch (e.type) {
            case Type::I32:
            case Type::I64:
            case Type::TS:
                decode_int_window(e, cc, oc, start, n, mut_, state_->zz,
                                  state_->dec, state_->slots);
                break;
            case Type::F64:
                decode_f64_window(e, cc, oc, start, n, mut_);
                break;
            case Type::BOOL:
                decode_bool_window(e, cc, oc, start, n);
                break;
            case Type::STR:  // WP-7b: STR out of compression grammar (mutant path)
                break;
        }
        out.add_column(std::move(oc));
    }
    cursor_ = start + n;
    current_ = std::move(out);
    return current_.view();
}

}  // namespace qe::mutant
