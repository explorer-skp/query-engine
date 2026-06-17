//  WP-14: time-series codecs + the compressed scan. See tsx/compress.h for the
//  framing, the lossless+nullable contract, and the decode-during-scan model.
//
//  Layout of the decode hot path (TS/I64/I32): VARINT byte-extract the window's
//  non-null words (sequential) -> ZIG-ZAG decode them (the vectorized kernel,
//  tsx/compress_kernels.h, selected by DecodePath) -> for delta-of-delta, a scalar
//  running double-prefix-sum (sequential) -> scatter into the non-null slots.
//  Gorilla (F64) is fully sequential and BOOL is a 1-bit unpack — both inline.

#include "tsx/compress.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>

#include "core/validity.h"
#include "tsx/compress_kernels.h"

namespace qe::tsx {
namespace {

// ---- bit / varint primitives (from scratch) --------------------------------

// MSB-first bit writer over a growing byte vector (Gorilla / BOOL encode).
struct BitWriter {
    std::vector<std::byte>& out;
    unsigned acc = 0;  // partial byte, MSB-first
    int nbits = 0;     // bits held in acc (0..7)

    void put_bit(unsigned b) {
        acc = (acc << 1) | (b & 1u);
        if (++nbits == 8) {
            out.push_back(static_cast<std::byte>(acc));
            acc = 0;
            nbits = 0;
        }
    }
    // Write the low `n` bits of `v`, most-significant first.
    void put_bits(std::uint64_t v, int n) {
        for (int i = n - 1; i >= 0; --i) put_bit(static_cast<unsigned>((v >> i) & 1u));
    }
    void flush() {
        if (nbits) {
            acc <<= (8 - nbits);  // left-justify the partial byte
            out.push_back(static_cast<std::byte>(acc));
            acc = 0;
            nbits = 0;
        }
    }
};

// MSB-first bit reader at an absolute bit position (so decode resumes across
// batches by carrying just the bit cursor).
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

void put_varint(std::vector<std::byte>& out, std::uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<std::byte>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<std::byte>(v));
}

// Varint-decode at an absolute byte position (advanced in place).
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

// Zig-zag of an int64 whose bits are `u` (pure unsigned; UBSan-clean).
std::uint64_t zigzag(std::uint64_t u) {
    return (u << 1) ^ (0ull - (u >> 63));
}

bool valid_bit(const EncodedColumn& e, std::size_t row) {
    if (e.all_valid) return true;
    return validity::get_bit(
        reinterpret_cast<const std::uint64_t*>(e.validity.data()), row);
}

// Read a value column at physical row r as the uint64 bit pattern of its stored
// scalar (I32 sign-extended; F64/I64/TS verbatim; BOOL as 0/1).
std::uint64_t read_bits(const OwnedColumn& oc, std::size_t r) {
    const std::byte* d = oc.data();
    switch (oc.type()) {
        case Type::I32: {
            const std::int64_t v = reinterpret_cast<const std::int32_t*>(d)[r];
            return static_cast<std::uint64_t>(v);  // sign-extend then reinterpret
        }
        case Type::I64:
        case Type::TS:
            return static_cast<std::uint64_t>(
                reinterpret_cast<const std::int64_t*>(d)[r]);
        case Type::F64: {
            std::uint64_t bits;
            std::memcpy(&bits, reinterpret_cast<const double*>(d) + r, 8);
            return bits;
        }
        case Type::BOOL:
            return reinterpret_cast<const std::uint8_t*>(d)[r] ? 1ull : 0ull;
    }
    return 0;
}

// ---- per-type ENCODERS ------------------------------------------------------

// Delta-of-delta + zig-zag + varint over the non-null int64-bit values (in order).
void encode_dod(const std::vector<std::uint64_t>& vals,
                std::vector<std::byte>& out) {
    std::uint64_t prev = 0, prev_delta = 0;
    for (std::uint64_t cur : vals) {
        const std::uint64_t delta = cur - prev;            // modular (wraps)
        const std::uint64_t dod = delta - prev_delta;      // second difference
        put_varint(out, zigzag(dod));
        prev_delta = delta;
        prev = cur;
    }
}

// Plain zig-zag + varint (I32, sign-extended into the uint64 zig-zag).
void encode_zigzag(const std::vector<std::uint64_t>& vals,
                   std::vector<std::byte>& out) {
    for (std::uint64_t u : vals) put_varint(out, zigzag(u));
}

// Gorilla XOR over the non-null F64 bit patterns (in order).
void encode_gorilla(const std::vector<std::uint64_t>& vals,
                    std::vector<std::byte>& out) {
    BitWriter bw{out};
    if (vals.empty()) {
        bw.flush();
        return;
    }
    std::uint64_t prev = vals[0];
    bw.put_bits(prev, 64);  // first value raw
    unsigned prev_lead = 0, prev_trail = 0, prev_mean = 0;
    bool have_window = false;
    for (std::size_t i = 1; i < vals.size(); ++i) {
        const std::uint64_t x = vals[i] ^ prev;
        prev = vals[i];
        if (x == 0) {
            bw.put_bit(0);
            continue;
        }
        bw.put_bit(1);
        const unsigned lead = static_cast<unsigned>(std::countl_zero(x));
        const unsigned trail = static_cast<unsigned>(std::countr_zero(x));
        if (have_window && lead >= prev_lead && trail >= prev_trail) {
            // Reuse the previous window: control bit 0, then prev_mean bits.
            bw.put_bit(0);
            bw.put_bits(x >> prev_trail, static_cast<int>(prev_mean));
        } else {
            // New window: control bit 1, 5-bit lead (clamped), 6-bit length.
            bw.put_bit(1);
            const unsigned L = lead > 31u ? 31u : lead;
            const unsigned mean = 64u - L - trail;  // 1..64
            bw.put_bits(L, 5);
            bw.put_bits(mean & 0x3Fu, 6);  // 64 stored as 0
            bw.put_bits(x >> trail, static_cast<int>(mean));
            prev_lead = L;
            prev_trail = trail;
            prev_mean = mean;
            have_window = true;
        }
    }
    bw.flush();
}

// 1-bit-per-value bit-pack of the non-null BOOL values.
void encode_bool(const std::vector<std::uint64_t>& vals,
                 std::vector<std::byte>& out) {
    BitWriter bw{out};
    for (std::uint64_t v : vals) bw.put_bit(static_cast<unsigned>(v & 1u));
    bw.flush();
}

EncodedColumn encode_column(const OwnedColumn& oc) {
    EncodedColumn e;
    e.type = oc.type();
    e.nrows = oc.len();
    e.all_valid = oc.all_valid();
    if (!e.all_valid) {
        const std::size_t nwords = validity::words(e.nrows);
        e.validity.resize(nwords * 8);
        std::memcpy(e.validity.data(), oc.validity(), nwords * 8);
    }
    // Collect the non-null values (bit patterns) in row order.
    std::vector<std::uint64_t> vals;
    vals.reserve(e.nrows);
    for (std::size_t r = 0; r < e.nrows; ++r) {
        const bool valid =
            e.all_valid || validity::get_bit(oc.validity(), r);
        if (valid) vals.push_back(read_bits(oc, r));
    }
    e.nonnull = vals.size();
    switch (e.type) {
        case Type::I64:
        case Type::TS:
            encode_dod(vals, e.values);
            break;
        case Type::I32:
            encode_zigzag(vals, e.values);
            break;
        case Type::F64:
            encode_gorilla(vals, e.values);
            break;
        case Type::BOOL:
            encode_bool(vals, e.values);
            break;
    }
    return e;
}

}  // namespace

// ---- CompressedTable --------------------------------------------------------

CompressedTable CompressedTable::encode(const Table& t) {
    CompressedTable ct;
    ct.schema_ = t.schema();
    ct.nrows_ = t.num_rows();
    ct.cols_.reserve(t.num_columns());
    for (std::size_t c = 0; c < t.num_columns(); ++c)
        ct.cols_.push_back(encode_column(t.column(c)));
    return ct;
}

std::size_t CompressedTable::compressed_bytes() const {
    std::size_t s = 0;
    for (const EncodedColumn& e : cols_) s += e.compressed_bytes();
    return s;
}

std::size_t CompressedTable::uncompressed_bytes(std::size_t i) const {
    return byte_width(cols_[i].type) * nrows_;
}

std::size_t CompressedTable::uncompressed_bytes() const {
    std::size_t s = 0;
    for (std::size_t i = 0; i < cols_.size(); ++i) s += uncompressed_bytes(i);
    return s;
}

// ---- CompressedScan ---------------------------------------------------------

namespace {

// Streaming decoder cursor for one column. Carries the stream position and the
// codec's running state across batch boundaries.
struct ColCursor {
    std::size_t byte_pos = 0;  // varint cursor (I32/I64/TS)
    std::size_t bit_pos = 0;   // Gorilla / BOOL cursor
    // delta-of-delta running state:
    std::uint64_t prev = 0;
    std::uint64_t prev_delta = 0;
    // Gorilla running state:
    std::uint64_t g_prev = 0;
    unsigned g_lead = 0;
    unsigned g_trail = 0;
    unsigned g_mean = 0;
    bool g_started = false;
};

}  // namespace

struct CompressedScan::State {
    std::vector<ColCursor> cur;       // one per column
    // Reusable per-decode scratch (sized to the window's non-null count).
    std::vector<std::uint64_t> zz;    // zig-zag words extracted by varint
    std::vector<std::uint64_t> dec;   // unzigzag kernel output
    std::vector<std::uint32_t> slots; // local indices of the non-null rows
};

CompressedScan::CompressedScan(const CompressedTable& ct, std::size_t batch_size)
    : ct_(ct), batch_size_(batch_size) {
    if (batch_size_ == 0)
        batch_size_ = Scan::kDefaultBatchSize;  // defensive; a leaf needs >=1 row
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

// Decode one window [start, start+n) of an integer column (I32/I64/TS) into `oc`,
// advancing `cc`. The zig-zag step runs the kernel selected by `path`.
void decode_int_window(const EncodedColumn& e, ColCursor& cc, OwnedColumn& oc,
                       std::size_t start, std::size_t n, DecodePath path,
                       std::vector<std::uint64_t>& zz,
                       std::vector<std::uint64_t>& dec,
                       std::vector<std::uint32_t>& slots) {
    const std::byte* vbytes = e.values.data();
    slots.clear();
    zz.clear();
    // 1) Varint byte-extract the window's non-null words (sequential).
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
    // 2) Zig-zag decode (the genuine vector/scalar twin step).
    if (m) {
        if (path == DecodePath::kVector)
            unzigzag_vec(zz.data(), m, dec.data());
        else
            unzigzag_scalar(zz.data(), m, dec.data());
    }
    auto* out = reinterpret_cast<std::byte*>(oc.mutable_data());
    const bool dod = (e.type == Type::I64 || e.type == Type::TS);
    // 3) Reconstruct + scatter into the non-null slots (delta-of-delta prefix is
    //    sequential and carries across batches via cc.prev / cc.prev_delta).
    for (std::size_t j = 0; j < m; ++j) {
        std::uint64_t bits;
        if (dod) {
            const std::uint64_t delta = cc.prev_delta + dec[j];
            const std::uint64_t val = cc.prev + delta;
            cc.prev_delta = delta;
            cc.prev = val;
            bits = val;
        } else {
            bits = dec[j];  // I32: zig-zag value bits directly
        }
        const std::uint32_t slot = slots[j];
        if (e.type == Type::I32)
            reinterpret_cast<std::int32_t*>(out)[slot] =
                static_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
        else
            std::memcpy(out + static_cast<std::size_t>(slot) * 8, &bits, 8);
    }
}

// Decode one window of a Gorilla F64 column (fully sequential).
void decode_f64_window(const EncodedColumn& e, ColCursor& cc, OwnedColumn& oc,
                       std::size_t start, std::size_t n) {
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
            value = cc.g_prev;  // XOR was zero
        } else {
            std::uint64_t x;
            if (br.get_bit() == 0) {  // reuse window
                const std::uint64_t meaningful =
                    br.get_bits(static_cast<int>(cc.g_mean));
                x = meaningful << cc.g_trail;
            } else {  // new window
                const unsigned L = static_cast<unsigned>(br.get_bits(5));
                const unsigned m6 = static_cast<unsigned>(br.get_bits(6));
                const unsigned mean = (m6 == 0u) ? 64u : m6;
                const std::uint64_t meaningful =
                    br.get_bits(static_cast<int>(mean));
                const unsigned trail = 64u - L - mean;
                x = meaningful << trail;
                cc.g_lead = L;
                cc.g_trail = trail;
                cc.g_mean = mean;
            }
            value = cc.g_prev ^ x;
        }
        cc.g_prev = value;
        std::memcpy(out + k, &value, 8);
    }
    cc.bit_pos = br.bitpos;
}

// Decode one window of a bit-packed BOOL column.
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
        // Zero the data buffer so a NULL slot's (semantically-undefined) bytes are
        // deterministic and never trip a reader of uninitialized memory.
        std::memset(oc.mutable_data(), 0, n * byte_width(e.type));
        ColCursor& cc = state_->cur[c];
        switch (e.type) {
            case Type::I32:
            case Type::I64:
            case Type::TS:
                decode_int_window(e, cc, oc, start, n, path_, state_->zz,
                                  state_->dec, state_->slots);
                break;
            case Type::F64:
                decode_f64_window(e, cc, oc, start, n);
                break;
            case Type::BOOL:
                decode_bool_window(e, cc, oc, start, n);
                break;
        }
        out.add_column(std::move(oc));
    }
    cursor_ = start + n;
    current_ = std::move(out);
    return current_.view();
}

}  // namespace qe::tsx
