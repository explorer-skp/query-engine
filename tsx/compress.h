//  WP-14: TIME-SERIES COMPRESSION + a compressed SCAN, the Phase-2 (tsx/) codec
//  component. From-scratch codecs (NO compression library — Gorilla, delta-of-
//  delta, zig-zag, varint are all hand-rolled here) plus a leaf operator that
//  DECODES batch-by-batch so the decompressed stream feeds the engine end-to-end.
//  THIS header is WP-14's OWN public surface (frozen at acceptance), mirroring the
//  small/explicit shape of tsx/asof.h and ops/scan.h + ops/table.h.
//
//  THE CODECS (one encode/decode pair per physical type so a whole Table round-
//  trips). Each column's stream is two independent parts: a VALIDITY stream (the
//  bitmap, byte-for-byte) and a VALUES stream holding ONLY the non-null values in
//  order, encoded per type:
//   * F64  -> GORILLA XOR. value bits XOR'd against the previous value; a zero XOR
//     is one bit, else a leading-zero / meaningful-bit-run block. Bit-exact across
//     the ENTIRE double domain because it never interprets the float — NaN payload
//     bits, ±0.0, ±inf and subnormals are preserved by construction.
//   * TS / I64 -> DELTA-OF-DELTA + ZIG-ZAG + VARINT. The second difference of the
//     timestamp series (computed in modular uint64 so it is well-defined and
//     reversible for EVERY int64 input — no signed overflow), zig-zagged to a small
//     unsigned magnitude, then LEB128 varint-coded.
//   * I32 -> ZIG-ZAG + VARINT (sign-extended through the same uint64 zig-zag, so the
//     decode shares one kernel; delta is intentionally omitted — kept simple).
//   * BOOL -> 1-bit-per-value BIT-PACK of the non-null values.
//
//  LOSSLESS + NULLABLE CONTRACT (the honest, testable definition). After
//  decode(encode(col)) the VALIDITY BITMAP is preserved EXACTLY (every null stays a
//  null, every valid stays valid) and every NON-NULL value is preserved BIT-EXACTLY
//  — including NaN payload bits, −0.0 distinct from +0.0, ±inf and subnormals (the
//  round-trip test compares the raw 8 bytes / bit pattern, never `==`). The bytes
//  occupying a NULL slot are SEMANTICALLY UNDEFINED and are NOT preserved: a NULL
//  carries no value, the validity bitmap already records its nullness, and Column
//  equality respects validity, so a null slot's data bytes are never read. Encoding
//  only the non-null values (and the bitmap) is therefore lossless in the only sense
//  that is observable — and it is what makes the compression honest (a column of all
//  nulls compresses to just its bitmap).
//
//  THE COMPRESSED SCAN (decode-during-scan, NOT decode-then-Scan). CompressedScan
//  holds a const CompressedTable& and, per next(), decodes the NEXT batch-window of
//  every column into a fresh OwnedBatch and yields its view — dense, <= batch_size
//  rows. Decoder state (bit/byte cursor + the running delta-of-delta / Gorilla
//  carry) persists across next() calls, so the stream is decoded sequentially
//  WITHOUT re-decoding from row 0 and the batch_size is free to vary (the §5
//  requirement that decode feeds the engine end-to-end). Because each batch
//  materializes fresh OwnedColumns with their own freshly-built validity bitmap,
//  there is no zero-copy word-alignment constraint (unlike ops/scan.h): batch_size
//  need only be positive.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/operator.h"
#include "ops/scan.h"   // Scan::kDefaultBatchSize
#include "ops/table.h"  // Table

namespace qe::tsx {

// Test seam (mirrors tsx::AsofJoin's GatherPath): selects the zig-zag-decode kernel
// the integer codecs run, so scalar==vector is a true end-to-end check. The
// inherently-sequential steps (varint extraction, delta-of-delta prefix, Gorilla
// reconstruction) are unaffected — they are scalar on both paths by construction.
enum class DecodePath { kVector, kScalar };

// One encoded column: the exactly-preserved validity stream + the codec'd values
// stream (non-null values only). `compressed_bytes()` is the honest on-the-wire
// size used for the ratio report.
struct EncodedColumn {
    Type type = Type::I32;
    std::size_t nrows = 0;     // total rows (incl. nulls)
    std::size_t nonnull = 0;   // count of non-null values actually encoded
    bool all_valid = true;     // true => no validity stream stored
    std::vector<std::byte> validity;  // words(nrows)*8 bytes when !all_valid
    std::vector<std::byte> values;    // codec bytes (varint / Gorilla bits / pack)

    std::size_t compressed_bytes() const { return validity.size() + values.size(); }
};

// Owns the ENCODED column buffers + the Schema. Constructed via encode(); it
// retains ONLY the encoded bytes (never the source values), so the scan genuinely
// decodes rather than handing back a cached copy.
class CompressedTable {
   public:
    // Encode every column of `t` (lossless per the header contract). Copies the
    // schema; does NOT borrow `t` afterwards.
    static CompressedTable encode(const Table& t);

    const Schema& schema() const noexcept { return schema_; }
    std::size_t num_rows() const noexcept { return nrows_; }
    std::size_t num_columns() const noexcept { return cols_.size(); }

    // The encoded stream for column i (what CompressedScan decodes from).
    const EncodedColumn& encoded(std::size_t i) const { return cols_[i]; }

    // On-the-wire compressed size (sum of every column's streams), and per column.
    std::size_t compressed_bytes() const;
    std::size_t compressed_bytes(std::size_t i) const {
        return cols_[i].compressed_bytes();
    }
    // The uncompressed in-memory size the engine would otherwise scan (data bytes
    // only; the denominator of the ratio report). Per type byte_width * nrows.
    std::size_t uncompressed_bytes() const;
    std::size_t uncompressed_bytes(std::size_t i) const;

   private:
    Schema schema_;
    std::size_t nrows_ = 0;
    std::vector<EncodedColumn> cols_;
};

// The leaf operator: streams `ct` out as dense <= batch_size-row Batches, decoding
// each column's next window per next(). Conforms to the frozen Operator contract.
class CompressedScan : public Operator {
   public:
    explicit CompressedScan(const CompressedTable& ct,
                            std::size_t batch_size = Scan::kDefaultBatchSize);

    // Test seam (call BEFORE open()): force the integer zig-zag-decode path. Default
    // kVector; downstream consumers ignore this.
    void set_path(DecodePath p) { path_ = p; }

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    const CompressedTable& ct_;
    std::size_t batch_size_;
    std::size_t cursor_ = 0;  // next row index to emit
    DecodePath path_ = DecodePath::kVector;

    // Per-column streaming decoder cursors + reusable scratch; opaque (lives in the
    // .cpp, like HashJoin/AsofJoin State).
    struct State;
    std::shared_ptr<State> state_;
    OwnedBatch current_;  // backs the view returned by next()
    bool opened_ = false;
};

}  // namespace qe::tsx
