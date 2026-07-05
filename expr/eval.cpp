//  WP-2: the tree-walking expression evaluator (see expr/expr.h).
//
//  evaluate() walks an Expr over a Batch and produces a DENSE OwnedColumn of
//  batch.row_count values. The batch's selection vector is applied when a Column
//  leaf is materialized (via WP-1 compact_column); every intermediate is dense,
//  so the op×type kernels (expr/kernels.h) see two same-length, same-type, dense
//  operands. VALUE computation is the kernels' job; NULL PROPAGATION is this
//  file's job — the truth table documented in expr/expr.h is implemented here:
//
//   * arithmetic / comparison : null-if-any (propagate_nulls_and); plus div/mod
//     by zero => NULL (null_where_divisor_zero).
//   * cast : input nulls propagate (copy_validity); out-of-range / non-finite
//     float->int and out-of-range int narrowing => NULL (null_where_cast_oor).
//   * logical : three-valued via a tri-state (0=F,1=N,2=T) round-trip, so the
//     value AND the validity follow Kleene rules (F∧N=F, T∨N=T, ...).

#include "expr/expr.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/types.h"
#include "core/validity.h"
#include "expr/cast_round.h"
#include "expr/eval_internal.h"
#include "expr/kernels.h"
#include "simd/validity_kernels.h"

namespace qe::expr {
namespace {

OwnedColumn eval_node(const Node& node, const Batch& batch, Backend be);

// ---- null-propagation helpers ----------------------------------------------

void copy_validity(OwnedColumn& out, const Column& in, std::size_t n) {
    if (in.all_valid || in.validity == nullptr) return;  // out stays all_valid
    out.ensure_validity();
    std::memcpy(out.mutable_validity(), in.validity,
                validity::words(n) * sizeof(std::uint64_t));
    out.refresh_all_valid();
}

bool divisor_is_zero(const OwnedColumn& b, Type t, std::size_t i) {
    switch (t) {
        case Type::I32:
            return reinterpret_cast<const std::int32_t*>(b.data())[i] == 0;
        case Type::I64:
        case Type::TS:
            return reinterpret_cast<const std::int64_t*>(b.data())[i] == 0;
        case Type::F64:
            return reinterpret_cast<const double*>(b.data())[i] == 0.0;
        default:
            return false;
    }
}

void null_where_divisor_zero(OwnedColumn& out, const OwnedColumn& b, Type t,
                             std::size_t n) {
    out.ensure_validity();
    std::uint64_t* ov = out.mutable_validity();
    for (std::size_t i = 0; i < n; ++i)
        if (divisor_is_zero(b, t, i)) validity::set_bit(ov, i, false);
    out.refresh_all_valid();
}

// True iff casting input value i (type `from`) to `to` is out of range / lossy
// enough to become NULL. Only F64->int and {I64,TS}->I32 can be out of range.
bool cast_lane_oor(const OwnedColumn& in, Type from, Type to, std::size_t i) {
    if (from == Type::F64 &&
        (to == Type::I32 || to == Type::I64 || to == Type::TS)) {
        const double v = reinterpret_cast<const double*>(in.data())[i];
        return !f64_fits_int(v, to);
    }
    if ((from == Type::I64 || from == Type::TS) && to == Type::I32) {
        const std::int64_t v = reinterpret_cast<const std::int64_t*>(in.data())[i];
        return v < -2147483648LL || v > 2147483647LL;
    }
    return false;
}

void null_where_cast_oor(OwnedColumn& out, const OwnedColumn& in, Type from,
                         Type to, std::size_t n) {
    if (!((from == Type::F64 &&
           (to == Type::I32 || to == Type::I64 || to == Type::TS)) ||
          ((from == Type::I64 || from == Type::TS) && to == Type::I32)))
        return;  // this cast can never go out of range
    out.ensure_validity();
    std::uint64_t* ov = out.mutable_validity();
    for (std::size_t i = 0; i < n; ++i)
        if (cast_lane_oor(in, from, to, i)) validity::set_bit(ov, i, false);
    out.refresh_all_valid();
}

// ---- leaf materialization --------------------------------------------------

OwnedColumn materialize_literal(const Scalar& s, std::size_t n) {
    OwnedColumn out = OwnedColumn::make(s.type, n);
    void* d = out.mutable_data();
    switch (s.type) {
        case Type::I32: {
            auto* p = static_cast<std::int32_t*>(d);
            for (std::size_t i = 0; i < n; ++i)
                p[i] = static_cast<std::int32_t>(s.i);
            break;
        }
        case Type::I64:
        case Type::TS: {
            auto* p = static_cast<std::int64_t*>(d);
            for (std::size_t i = 0; i < n; ++i) p[i] = s.i;
            break;
        }
        case Type::F64: {
            auto* p = static_cast<double*>(d);
            for (std::size_t i = 0; i < n; ++i) p[i] = s.f;
            break;
        }
        case Type::BOOL: {
            auto* p = static_cast<std::uint8_t*>(d);
            for (std::size_t i = 0; i < n; ++i)
                p[i] = static_cast<std::uint8_t>(s.i ? 1 : 0);
            break;
        }
        case Type::STR:
            // WP-7b: the frozen Scalar (expr/expr.h) carries no string payload, so
            // a STR literal cannot be represented and the builders never make one.
            // STR comparisons in this engine are column-vs-column (see the WP
            // report). Reject loudly rather than fabricate a value.
            throw std::invalid_argument(
                "STR literals are unsupported (Scalar has no string payload); "
                "compare STR columns to STR columns");
    }
    if (s.is_null && n > 0) {
        out.ensure_validity();
        validity::fill_all_null(out.mutable_validity(), n);
        out.refresh_all_valid();
    }
    return out;
}

// ---- tri-state round-trip for three-valued logic ---------------------------

std::vector<std::uint8_t> to_tristate(const Column& c, std::size_t n) {
    std::vector<std::uint8_t> ts(n);
    const auto* v = reinterpret_cast<const std::uint8_t*>(c.data);
    for (std::size_t i = 0; i < n; ++i) {
        const bool valid =
            c.all_valid || c.validity == nullptr || validity::get_bit(c.validity, i);
        ts[i] = valid ? (v[i] ? 2 : 0) : 1;  // 0=F,1=N,2=T
    }
    return ts;
}

OwnedColumn from_tristate(const std::vector<std::uint8_t>& ts, std::size_t n) {
    OwnedColumn out = OwnedColumn::make(Type::BOOL, n);
    auto* d = reinterpret_cast<std::uint8_t*>(out.mutable_data());
    bool any_null = false;
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = (ts[i] == 2) ? 1 : 0;
        if (ts[i] == 1) any_null = true;
    }
    if (any_null) {
        out.ensure_validity();
        std::uint64_t* ov = out.mutable_validity();
        for (std::size_t i = 0; i < n; ++i)
            if (ts[i] == 1) validity::set_bit(ov, i, false);
        out.refresh_all_valid();
    }
    return out;
}

// ---- STR comparison (by VALUE via the dicts) -------------------------------
//
// WP-7b: STR operands are dictionary CODES; equality/order is by the resolved
// string BYTES, never the raw code (two columns may carry different dicts, and
// equal strings can get different codes). This is inherently SCALAR control flow
// (variable-length byte compare), so there is no Highway twin — the Vector and
// Scalar backends drive this same routine (documented in the WP report; the
// scalar==vector gate over STR is therefore trivially satisfied because the STR
// path is identical on both backends, exactly the WP-5/WP-6/WP-12 precedent for
// inherently-sequential steps). `a`/`b` are dense (selection already applied);
// codes index directly. DuckDB VARCHAR ordering is bytewise on ASCII text, which
// is std::string_view::compare.
void cmp_str(CmpOp op, const Column& a, const Column& b, std::uint8_t* out,
             std::size_t n) {
    const auto* ca = reinterpret_cast<const std::int32_t*>(a.data);
    const auto* cb = reinterpret_cast<const std::int32_t*>(b.data);
    for (std::size_t i = 0; i < n; ++i) {
        // NULL lanes carry NO defined code (builders only set_null them), so a
        // dict lookup there is UB (unchecked offsets_ indexing — audit H3).
        // propagate_nulls_and masks these lanes afterwards; emit a dummy byte.
        const bool va = a.all_valid || validity::get_bit(a.validity, i);
        const bool vb = b.all_valid || validity::get_bit(b.validity, i);
        if (!va || !vb) {
            out[i] = 0;  // masked by the null-propagation pass
            continue;
        }
        const int c = a.dict->at(ca[i]).compare(b.dict->at(cb[i]));
        bool r = false;
        switch (op) {
            case CmpOp::Lt: r = c < 0; break;
            case CmpOp::Le: r = c <= 0; break;
            case CmpOp::Gt: r = c > 0; break;
            case CmpOp::Ge: r = c >= 0; break;
            case CmpOp::Eq: r = c == 0; break;
            case CmpOp::Ne: r = c != 0; break;
        }
        out[i] = r ? 1 : 0;
    }
}

// ---- node evaluation -------------------------------------------------------

OwnedColumn eval_binary_arith(const Node& node, const Batch& batch, Backend be,
                              std::size_t n) {
    OwnedColumn a = eval_node(node.children[0].node(), batch, be);
    OwnedColumn b = eval_node(node.children[1].node(), batch, be);
    OwnedColumn out = OwnedColumn::make(node.type, n);
    if (be == Backend::Vector)
        arith_vec(node.arith_op, node.type, a.data(), b.data(),
                  out.mutable_data(), n);
    else
        arith_scalar(node.arith_op, node.type, a.data(), b.data(),
                     out.mutable_data(), n);
    propagate_nulls_and(out, a.view(), b.view());
    if (node.arith_op == ArithOp::Div || node.arith_op == ArithOp::Mod)
        null_where_divisor_zero(out, b, node.type, n);
    return out;
}

OwnedColumn eval_cmp(const Node& node, const Batch& batch, Backend be,
                     std::size_t n) {
    OwnedColumn a = eval_node(node.children[0].node(), batch, be);
    OwnedColumn b = eval_node(node.children[1].node(), batch, be);
    const Type opnd = node.children[0].type();  // both children share this type
    OwnedColumn out = OwnedColumn::make(Type::BOOL, n);
    auto* o = reinterpret_cast<std::uint8_t*>(out.mutable_data());
    if (opnd == Type::STR)
        cmp_str(node.cmp_op, a.view(), b.view(), o, n);  // by value (both backends)
    else if (be == Backend::Vector)
        cmp_vec(node.cmp_op, opnd, a.data(), b.data(), o, n);
    else
        cmp_scalar(node.cmp_op, opnd, a.data(), b.data(), o, n);
    propagate_nulls_and(out, a.view(), b.view());
    return out;
}

OwnedColumn eval_logic(const Node& node, const Batch& batch, Backend be,
                       std::size_t n) {
    if (node.logic_op == LogicOp::Not) {
        OwnedColumn a = eval_node(node.children[0].node(), batch, be);
        std::vector<std::uint8_t> ta = to_tristate(a.view(), n);
        std::vector<std::uint8_t> to(n);
        if (be == Backend::Vector)
            logic_not_vec(ta.data(), to.data(), n);
        else
            logic_not_scalar(ta.data(), to.data(), n);
        return from_tristate(to, n);
    }
    OwnedColumn a = eval_node(node.children[0].node(), batch, be);
    OwnedColumn b = eval_node(node.children[1].node(), batch, be);
    std::vector<std::uint8_t> ta = to_tristate(a.view(), n);
    std::vector<std::uint8_t> tb = to_tristate(b.view(), n);
    std::vector<std::uint8_t> to(n);
    if (node.logic_op == LogicOp::And) {
        if (be == Backend::Vector)
            logic_and_vec(ta.data(), tb.data(), to.data(), n);
        else
            logic_and_scalar(ta.data(), tb.data(), to.data(), n);
    } else {
        if (be == Backend::Vector)
            logic_or_vec(ta.data(), tb.data(), to.data(), n);
        else
            logic_or_scalar(ta.data(), tb.data(), to.data(), n);
    }
    return from_tristate(to, n);
}

OwnedColumn eval_cast(const Node& node, const Batch& batch, Backend be,
                      std::size_t n) {
    const Type from = node.children[0].type();
    const Type to = node.cast_to;
    OwnedColumn in = eval_node(node.children[0].node(), batch, be);
    OwnedColumn out = OwnedColumn::make(to, n);
    if (be == Backend::Vector)
        cast_vec(from, to, in.data(), out.mutable_data(), n);
    else
        cast_scalar(from, to, in.data(), out.mutable_data(), n);
    copy_validity(out, in.view(), n);
    null_where_cast_oor(out, in, from, to, n);
    return out;
}

OwnedColumn eval_node(const Node& node, const Batch& batch, Backend be) {
    const std::size_t n = batch.row_count;
    switch (node.kind) {
        case NodeKind::Column:
            return compact_column(batch.cols[node.col_index], batch.sel, n);
        case NodeKind::Literal:
            return materialize_literal(node.literal, n);
        case NodeKind::Arith:
            return eval_binary_arith(node, batch, be, n);
        case NodeKind::Cmp:
            return eval_cmp(node, batch, be, n);
        case NodeKind::Logic:
            return eval_logic(node, batch, be, n);
        case NodeKind::Cast:
            return eval_cast(node, batch, be, n);
    }
    return OwnedColumn::make(node.type, n);  // unreachable
}

}  // namespace

// Exposed in expr/eval_internal.h (shared with the mutation self-test).
void propagate_nulls_and(OwnedColumn& out, const Column& a, const Column& b) {
    const std::size_t n = out.len();
    if (a.all_valid && b.all_valid) return;  // both clean => out stays all_valid
    out.ensure_validity();
    std::uint64_t* ov = out.mutable_validity();
    const std::size_t w = validity::words(n);
    if (a.all_valid || a.validity == nullptr) {
        std::memcpy(ov, b.validity, w * sizeof(std::uint64_t));
    } else if (b.all_valid || b.validity == nullptr) {
        std::memcpy(ov, a.validity, w * sizeof(std::uint64_t));
    } else {
        simd::bit_and_vec(a.validity, b.validity, ov, n);
    }
    out.refresh_all_valid();
}

OwnedColumn evaluate(const Expr& e, const Batch& batch, Backend backend) {
    return eval_node(e.node(), batch, backend);
}

}  // namespace qe::expr
