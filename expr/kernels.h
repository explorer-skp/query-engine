//  WP-2: INTERNAL kernel dispatch surface for expression evaluation.
//
//  NOT a frozen public contract — this is the seam between eval.cpp and the
//  op×type kernel matrix. Each family exposes two functions with identical
//  signatures following the twin convention (simd/kernel_convention.h):
//    *_vec    — Highway dynamic-dispatch path   (expr/<family>_kernels.cpp)
//    *_scalar — independently-written reference  (expr/<family>_scalar.cpp)
//  They MUST agree on every input; tests/expr_scalar_vector_test.cpp proves it.
//
//  D17: the vector matrix is one templated body per op (typed by lane T), the
//  scalar matrix is a SEPARATE templated body — same shape, different code — so
//  `scalar == vector` compares two independent implementations, not one.
//
//  VALUES ONLY. These kernels compute result VALUES across all n lanes and write
//  a DEFINED value to every lane (UB-free even where the result will be NULL).
//  NULL propagation (which lanes are NULL, including div-by-zero and out-of-range
//  cast) is owned by eval.cpp — see expr/eval_internal.h.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/types.h"
#include "expr/expr.h"

namespace qe::expr {

// Arithmetic. a, b, out are n values of physical type `t` (I32/I64/F64 only).
// Integer +,-,* wrap (two's complement); /,% guard a zero or INT_MIN/-1 divisor
// to a defined value (eval NULLs those lanes). Float is IEEE-754.
void arith_vec(ArithOp op, Type t, const void* a, const void* b, void* out,
               std::size_t n);
void arith_scalar(ArithOp op, Type t, const void* a, const void* b, void* out,
                  std::size_t n);

// Comparison. a, b are n values of physical type `t` (any of the five Types);
// out is n bytes, each 0 or 1.
void cmp_vec(CmpOp op, Type t, const void* a, const void* b, std::uint8_t* out,
             std::size_t n);
void cmp_scalar(CmpOp op, Type t, const void* a, const void* b,
                std::uint8_t* out, std::size_t n);

// Three-valued logic over TRI-STATE bytes (0 = FALSE, 1 = NULL, 2 = TRUE). The
// encoding makes Kleene AND = min, OR = max, NOT = 2 - x; eval converts a BOOL
// column's (value bytes + validity bitmap) to/from this encoding.
void logic_and_vec(const std::uint8_t* a, const std::uint8_t* b,
                   std::uint8_t* out, std::size_t n);
void logic_and_scalar(const std::uint8_t* a, const std::uint8_t* b,
                      std::uint8_t* out, std::size_t n);
void logic_or_vec(const std::uint8_t* a, const std::uint8_t* b,
                  std::uint8_t* out, std::size_t n);
void logic_or_scalar(const std::uint8_t* a, const std::uint8_t* b,
                     std::uint8_t* out, std::size_t n);
void logic_not_vec(const std::uint8_t* a, std::uint8_t* out, std::size_t n);
void logic_not_scalar(const std::uint8_t* a, std::uint8_t* out, std::size_t n);

// Cast. in is n values of physical type `from`, out is n values of `to`. Writes
// a defined value to every lane (eval NULLs out-of-range / non-finite lanes).
void cast_vec(Type from, Type to, const void* in, void* out, std::size_t n);
void cast_scalar(Type from, Type to, const void* in, void* out, std::size_t n);

}  // namespace qe::expr
