//  WP-2: SCALAR TWIN for the arithmetic kernels (see expr/kernels.h).
//
//  Plain, Highway-free C++. Independently written from the vector path
//  (expr/arith_kernels.cpp) so `scalar == vector` compares two implementations.
//
//  UB-free integer policy (RIGOR.md / the overflow policy in expr/expr.h):
//   +,-,*  computed through unsigned intermediates (two's-complement wraparound,
//          well-defined in C++20). /,%  guard a zero divisor (defined dummy value
//          — eval NULLs the lane) and the INT_MIN/-1 overflow (wrapping negate).
//   float  IEEE-754; % is std::fmod.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core/types.h"
#include "expr/expr.h"
#include "expr/kernels.h"

namespace qe::expr {
namespace {

template <typename T>
T wrap_add(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}
template <typename T>
T wrap_sub(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}
template <typename T>
T wrap_mul(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}
template <typename T>
T wrap_neg(T a) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(U{0} - static_cast<U>(a));
}

template <typename T>
void arith_int(ArithOp op, const T* a, const T* b, T* out, std::size_t n) {
    switch (op) {
        case ArithOp::Add:
            for (std::size_t i = 0; i < n; ++i) out[i] = wrap_add(a[i], b[i]);
            break;
        case ArithOp::Sub:
            for (std::size_t i = 0; i < n; ++i) out[i] = wrap_sub(a[i], b[i]);
            break;
        case ArithOp::Mul:
            for (std::size_t i = 0; i < n; ++i) out[i] = wrap_mul(a[i], b[i]);
            break;
        case ArithOp::Div:
            for (std::size_t i = 0; i < n; ++i) {
                if (b[i] == 0) {
                    out[i] = 0;  // NULL'd by eval; defined here to avoid UB
                } else if (b[i] == static_cast<T>(-1)) {
                    out[i] = wrap_neg(a[i]);  // INT_MIN/-1 wraps to INT_MIN
                } else {
                    out[i] = static_cast<T>(a[i] / b[i]);
                }
            }
            break;
        case ArithOp::Mod:
            for (std::size_t i = 0; i < n; ++i) {
                if (b[i] == 0 || b[i] == static_cast<T>(-1)) {
                    out[i] = 0;  // x % -1 == 0; b==0 NULL'd by eval
                } else {
                    out[i] = static_cast<T>(a[i] % b[i]);
                }
            }
            break;
    }
}

void arith_f64(ArithOp op, const double* a, const double* b, double* out,
               std::size_t n) {
    switch (op) {
        case ArithOp::Add:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
            break;
        case ArithOp::Sub:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
            break;
        case ArithOp::Mul:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
            break;
        case ArithOp::Div:
            for (std::size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
            break;
        case ArithOp::Mod:
            for (std::size_t i = 0; i < n; ++i) out[i] = std::fmod(a[i], b[i]);
            break;
    }
}

}  // namespace

void arith_scalar(ArithOp op, Type t, const void* a, const void* b, void* out,
                  std::size_t n) {
    switch (t) {
        case Type::I32:
            arith_int<std::int32_t>(op, static_cast<const std::int32_t*>(a),
                                    static_cast<const std::int32_t*>(b),
                                    static_cast<std::int32_t*>(out), n);
            break;
        case Type::I64:
            arith_int<std::int64_t>(op, static_cast<const std::int64_t*>(a),
                                    static_cast<const std::int64_t*>(b),
                                    static_cast<std::int64_t*>(out), n);
            break;
        case Type::F64:
            arith_f64(op, static_cast<const double*>(a),
                      static_cast<const double*>(b),
                      static_cast<double*>(out), n);
            break;
        case Type::BOOL:
        case Type::TS:
        case Type::STR:
            // Builder never produces arithmetic on these (STR arith is a build-time
            // type error in expr.cpp::arith; TS/BOOL require an explicit cast).
            break;
    }
}

}  // namespace qe::expr
