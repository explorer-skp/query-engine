//  WP-2: SCALAR TWIN for the cast kernels (see expr/kernels.h).
//
//  Plain, Highway-free C++, independent of expr/cast_kernels.cpp. Computes a
//  DEFINED value at every lane (the float->int range guard prevents UB); eval
//  applies the NULL for out-of-range / non-finite lanes using the same predicate
//  (expr/cast_round.h), so kernel value and eval NULL never disagree.
//
//  Semantics: numeric<->numeric standard conversion (float->int rounds half away
//  from zero); ->BOOL is (value != 0); BOOL->numeric is {0,1}; TS is int64-ns so
//  TS<->I64 is a reinterpret and TS participates in int conversions as int64.

#include <cstddef>
#include <cstdint>

#include "core/types.h"
#include "expr/cast_round.h"
#include "expr/kernels.h"

namespace qe::expr {
namespace {

// Read element i of an integer-family source (I32/I64/TS/BOOL) as int64.
std::int64_t read_int(Type from, const void* in, std::size_t i) {
    switch (from) {
        case Type::I32:
            return static_cast<const std::int32_t*>(in)[i];
        case Type::I64:
        case Type::TS:
            return static_cast<const std::int64_t*>(in)[i];
        case Type::BOOL:
            return static_cast<const std::uint8_t*>(in)[i];
        default:
            return 0;  // F64 handled by the caller
    }
}

// Write int64 v into element i of an integer-family target (I32/I64/TS).
void write_int(Type to, void* out, std::size_t i, std::int64_t v) {
    switch (to) {
        case Type::I32:
            static_cast<std::int32_t*>(out)[i] = static_cast<std::int32_t>(v);
            break;
        case Type::I64:
        case Type::TS:
            static_cast<std::int64_t*>(out)[i] = v;
            break;
        default:
            break;
    }
}

}  // namespace

void cast_scalar(Type from, Type to, const void* in, void* out, std::size_t n) {
    const double* ind = static_cast<const double*>(in);
    for (std::size_t i = 0; i < n; ++i) {
        if (to == Type::F64) {
            const double v =
                (from == Type::F64) ? ind[i]
                                    : static_cast<double>(read_int(from, in, i));
            static_cast<double*>(out)[i] = v;
        } else if (to == Type::BOOL) {
            const bool b = (from == Type::F64) ? (ind[i] != 0.0)
                                               : (read_int(from, in, i) != 0);
            static_cast<std::uint8_t*>(out)[i] = b ? 1 : 0;
        } else {  // integer target (I32 / I64 / TS)
            if (from == Type::F64) {
                if (f64_fits_int(ind[i], to)) {
                    write_int(to, out, i,
                              static_cast<std::int64_t>(cast_round(ind[i])));
                } else {
                    write_int(to, out, i, 0);  // NULL'd by eval
                }
            } else {
                write_int(to, out, i, read_int(from, in, i));
            }
        }
    }
}

}  // namespace qe::expr
