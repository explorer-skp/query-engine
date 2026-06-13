//  DRAFT — NOT FROZEN. Owned by the final review. Frozen at WP-1 (core/simd) /
//  WP-3 (Operator). Do not add fields, rename, or implement logic here.
//
//  WP-0 scaffold: this is a non-binding signature stub. The engine's value
//  vocabulary. Everything downstream is a typed column batch of one of these.
#pragma once

namespace qe {

enum class Type { I32, I64, F64, BOOL, TS };  // TS = int64 ns since epoch

}  // namespace qe
