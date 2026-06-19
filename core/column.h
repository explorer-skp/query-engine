//  Frozen at WP-1 (core/simd). The typed column batch that flows through every
//  pull-based vectorized operator (~2048-value batches).
//
//  These structs are NON-OWNING VIEWS — the contract every operator passes by
//  value. The bytes they point at are owned elsewhere (see core/buffer.h for the
//  owning Buffer and core/owned_batch.h for the OwnedColumn/OwnedBatch builders
//  that produce these views). An operator receives views, reads through them,
//  and produces new owned storage whose views it returns; it never frees through
//  a Column/Batch.
//
//  CONTRACT (frozen): the PUBLIC FIELDS of Column, SelectionVector, Batch, and
//  Schema below are the contract. Later WPs may not add, rename, or retype a
//  field without an Interface Change Request. Helper *types* live in sibling
//  headers; helper *functions* may be added.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"

namespace qe {

// Forward declaration: the out-of-band code->bytes table for Type::STR columns.
// Defined in the sibling helper header core/string_dict.h (a helper TYPE, which
// the frozen contract permits). The Column view only needs the pointer.
class StringDict;

// A view over one contiguous, type-width column of `len` values.
//
//   data:     `len * byte_width(type)` contiguous bytes (see core/types.h for
//             the per-type layout; BOOL is one uint8_t per value, not bit-packed).
//   validity: Arrow-style 1-bit-per-value bitmap, 64-bit little-endian words,
//             bit i of word i/64 (LSB-first); 1 == valid, 0 == null. Covers
//             `len` bits; bits past `len` in the final partial word are padding
//             and MUST NOT be read by consumers. May be nullptr (see below).
//   all_valid: the per-column fast-path flag (decision D5).
//
//  VALIDITY PRECEDENCE (frozen at WP-1 — the rule every consumer obeys):
//   1. `all_valid == true`  => the column has NO nulls. `validity` is IGNORED and
//      may be nullptr (canonical) or point at an all-ones bitmap (allowed but
//      redundant). This is the fast path: skip the bitmap entirely.
//   2. `all_valid == false` && `validity != nullptr` => per-value nullness is
//      read from the bitmap.
//   3. `validity == nullptr` IMPLIES `all_valid == true`. The state
//      (all_valid == false, validity == nullptr) is INVALID — a column claiming
//      nulls while supplying no bitmap — and is a precondition violation
//      (asserted in debug builds; see core/owned_batch.h which never produces it).
//
//  In one line: `all_valid` is authoritative; consult `validity` only when
//  `all_valid` is false.
struct Column {
    Type type;
    size_t len;
    const std::byte* data;     // contiguous, type-width
    const uint64_t* validity;  // nullptr => no nulls present (implies all_valid)
    bool all_valid;            // fast-path flag; true => no nulls, ignore validity
    // WP-7b (pre-authorized ICR-2, additive field appended after all_valid —
    // existing fields byte-unchanged). Non-null IFF type==STR: resolves the int32
    // dictionary codes in `data` to their string bytes. nullptr for every non-STR
    // column. The pointed-at StringDict is owned elsewhere (the source OwnedColumn
    // / operator state) and must outlive this view, exactly like `data`/`validity`.
    const StringDict* dict;
};

// Optional per-batch selection vector (decision D6): an ordered list of physical
// row indices that are "live". `idx[k]` is the physical row of the k-th logical
// row; indices are into the underlying columns' value space, 0 <= idx[k] < len.
// A Batch with `sel == nullptr` is dense (logical row k == physical row k).
struct SelectionVector {
    const uint32_t* idx;
    size_t len;
};

// A batch of equal-length column views plus an optional selection vector. Like
// Column, this is a VIEW: it does not own `cols`' bytes nor `sel`. `row_count`
// is the number of LOGICAL rows — it equals `sel->len` when `sel != nullptr`,
// otherwise the columns' physical length.
struct Batch {
    std::vector<Column> cols;
    const SelectionVector* sel;  // nullptr => dense 0..len-1
    size_t row_count;            // logical rows (respects sel)
};

struct Schema {
    std::vector<std::pair<std::string, Type>> fields;
};

}  // namespace qe
