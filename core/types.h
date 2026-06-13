//  Frozen at WP-1 (core/simd). The engine's value vocabulary. Everything
//  downstream is a typed column batch of one of these.
//
//  CONTRACT (frozen): the `Type` enumerators and their meaning are the contract.
//  Later work packages dispatch on these and must not add, rename, or renumber
//  them without an Interface Change Request.
#pragma once

#include <cstddef>

namespace qe {

enum class Type { I32, I64, F64, BOOL, TS };  // TS = int64 ns since epoch

// Physical width, in bytes, of one stored value of `t` in a Column's `data`
// buffer. This fixes the in-memory layout every later WP relies on:
//
//   I32  -> 4   (int32_t)
//   I64  -> 8   (int64_t)
//   F64  -> 8   (double)
//   TS   -> 8   (int64_t ns since epoch)
//   BOOL -> 1   (one uint8_t per value, 0 or 1 — NOT bit-packed; the validity
//                bitmap is the only bit-packed structure in the format)
//
// Added at WP-1 as a free function (it introduces no field on `Type`); it is the
// single source of truth for value width so no caller hardcodes a literal.
inline constexpr std::size_t byte_width(Type t) {
    switch (t) {
        case Type::I32:
            return 4;
        case Type::I64:
            return 8;
        case Type::F64:
            return 8;
        case Type::BOOL:
            return 1;
        case Type::TS:
            return 8;
    }
    return 0;  // unreachable; all enumerators handled above
}

}  // namespace qe
