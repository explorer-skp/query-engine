//  WP-2: SCALAR TWIN for the comparison kernels (see expr/kernels.h).
//
//  Plain, Highway-free C++, independent of expr/compare_kernels.cpp. Output is
//  one byte per value (0/1). Float comparisons follow IEEE-754 (any comparison
//  involving NaN is false, except != which is true) — identical to C++ operators,
//  which is what the vector path's Highway compares also implement.

#include <cstddef>
#include <cstdint>

#include "core/types.h"
#include "expr/expr.h"
#include "expr/kernels.h"

namespace qe::expr {
namespace {

template <typename T>
void cmp_t(CmpOp op, const T* a, const T* b, std::uint8_t* out, std::size_t n) {
    switch (op) {
        case CmpOp::Lt:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] < b[i] ? 1 : 0;
            break;
        case CmpOp::Le:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] <= b[i] ? 1 : 0;
            break;
        case CmpOp::Gt:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] > b[i] ? 1 : 0;
            break;
        case CmpOp::Ge:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] >= b[i] ? 1 : 0;
            break;
        case CmpOp::Eq:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] == b[i] ? 1 : 0;
            break;
        case CmpOp::Ne:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] != b[i] ? 1 : 0;
            break;
    }
}

}  // namespace

void cmp_scalar(CmpOp op, Type t, const void* a, const void* b,
                std::uint8_t* out, std::size_t n) {
    switch (t) {
        case Type::I32:
            cmp_t<std::int32_t>(op, static_cast<const std::int32_t*>(a),
                                static_cast<const std::int32_t*>(b), out, n);
            break;
        case Type::I64:
        case Type::TS:
            cmp_t<std::int64_t>(op, static_cast<const std::int64_t*>(a),
                                static_cast<const std::int64_t*>(b), out, n);
            break;
        case Type::F64:
            cmp_t<double>(op, static_cast<const double*>(a),
                          static_cast<const double*>(b), out, n);
            break;
        case Type::BOOL:
            cmp_t<std::uint8_t>(op, static_cast<const std::uint8_t*>(a),
                                static_cast<const std::uint8_t*>(b), out, n);
            break;
    }
}

}  // namespace qe::expr
