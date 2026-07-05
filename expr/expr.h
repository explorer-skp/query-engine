//  WP-2: Expression IR + evaluation entry point — THE FROZEN CONTRACT.
//
//  On acceptance this header is frozen: WP-3 (filter/project) and WP-8 (plan)
//  build expression trees with these factories and evaluate them with
//  evaluate(). Per the interface discipline (RIGOR.md), the PUBLIC SURFACE
//  below — the enums, Scalar, Expr, the builder free functions, and evaluate()
//  — is the contract; downstream WPs may not change a signature without an ICR.
//
//  THE MODEL
//   * An Expr is an immutable, shared, typed tree node (column ref, literal,
//     arithmetic, comparison, three-valued logical, or cast). Trees are built
//     with the free-function builders below, which perform TYPE INFERENCE and
//     VALIDATION at build time (throwing std::invalid_argument on a type error)
//     and AUTO-INSERT numeric-promotion casts so every binary kernel sees two
//     operands of one identical physical type. Expr::type() is the inferred
//     result Type.
//   * evaluate(expr, batch) walks the tree and produces a DENSE OwnedColumn of
//     batch.row_count values (the batch's selection vector is applied during
//     evaluation; the result is never selected). Results are allocated through
//     core/owned_batch.h (OwnedColumn), the engine's owning layer.
//
//  NULL PROPAGATION (the truth table this engine guarantees; tested
//  exhaustively in tests/expr_null_truthtable_test.cpp; aligned with DuckDB so
//  WP-3's differential passes):
//
//   arithmetic (+ - * / %) and comparison (< <= > >= == !=):
//       result is NULL iff ANY operand is NULL ("null-if-any").
//   division / modulo by zero (integer AND float):
//       result is NULL. Integer /0 -> NULL matches DuckDB (which raises only
//       outside SQL-NULL mode; the oracle renders `//`). Float /0 -> NULL is a
//       DOCUMENTED DIVERGENCE from the pinned DuckDB v1.1.3 default
//       (ieee_floating_point_ops=true -> ±inf/NaN); division is excluded from
//       the oracle grammar (oracle/generators.h), so neither behavior is
//       differentially checked. Also keeps integer division UB-free — see the
//       overflow policy below.
//   NaN in comparisons: IEEE semantics (every comparison with NaN is FALSE
//       except !=). DuckDB instead defines NaN = NaN as TRUE and orders NaN
//       greater than all values — a DOCUMENTED DIVERGENCE, masked by the
//       finite-only generator domain (aggregation/sort NaN ordering, by
//       contrast, IS aligned with DuckDB and differentially tested).
//   logical AND / OR / NOT — SQL/Kleene THREE-VALUED logic:
//       AND:  T∧T=T  T∧F=F  T∧N=N   F∧F=F  F∧N=F   N∧N=N
//       OR:   T∨T=T  T∨F=T  T∨N=T   F∨F=F  F∨N=N   N∨N=N
//       NOT:  ¬T=F   ¬F=T   ¬N=N
//     (N = NULL. Note F∧N=F and T∨N=T — a known value can dominate a NULL; a
//      plain "null-if-any" rule is WRONG here and is a planted-bug hotspot.)
//   cast:
//       NULL propagates (NULL in => NULL out). A value that is out of the target
//       integer range, or a non-finite float cast to an integer, yields NULL
//       (DuckDB raises; we model the failure as NULL — see cast notes in the WP
//       report). float->int rounds half away from zero (DuckDB semantics).
//
//  ARITHMETIC OVERFLOW POLICY (documented; UB-free so UBSan stays green):
//   signed integer +,-,*  : two's-complement WRAPAROUND. Computed through
//       unsigned intermediates (well-defined in C++20, which mandates two's
//       complement); the SIMD path uses Highway's non-saturating ops, which wrap
//       at the hardware level. No C++ signed-overflow UB is ever executed.
//   signed integer /, %   : divisor 0 => NULL (never divides by zero); the
//       INT_MIN / -1 (and INT_MIN % -1) overflow case is computed via wrapping
//       negation, yielding INT_MIN (and 0), never UB.
//   float                 : IEEE-754 (overflow => ±inf, invalid => NaN), except
//       /0 and %0 which this engine maps to NULL (above).
#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"

namespace qe::expr {

// ---- operator vocabularies -------------------------------------------------

enum class ArithOp { Add, Sub, Mul, Div, Mod };
enum class CmpOp { Lt, Le, Gt, Ge, Eq, Ne };
enum class LogicOp { And, Or, Not };  // And/Or binary; Not unary

// ---- typed scalar literal --------------------------------------------------

// One literal value of exactly one Type. is_null marks a typed SQL NULL. The
// payload is carried in `i` for the integer-family types (I32 in int32 range,
// I64, TS = int64 ns, BOOL as 0/1) and in `f` for F64; the other is ignored.
struct Scalar {
    Type type = Type::I32;
    bool is_null = false;
    std::int64_t i = 0;
    double f = 0.0;

    static Scalar i32(std::int32_t v) {
        return {Type::I32, false, v, 0.0};
    }
    static Scalar i64(std::int64_t v) { return {Type::I64, false, v, 0.0}; }
    static Scalar f64(double v) { return {Type::F64, false, 0, v}; }
    static Scalar boolean(bool v) {
        return {Type::BOOL, false, v ? 1 : 0, 0.0};
    }
    static Scalar ts(std::int64_t ns) { return {Type::TS, false, ns, 0.0}; }
    static Scalar null(Type t) {
        Scalar s;
        s.type = t;
        s.is_null = true;
        return s;
    }
};

// ---- the expression node ---------------------------------------------------

struct Node;  // defined below

// An Expr is a value-semantics handle to an immutable shared Node. Copying an
// Expr is cheap (shared_ptr) and trees freely share subexpressions.
class Expr {
   public:
    Expr() = default;
    explicit Expr(std::shared_ptr<const Node> n) : node_(std::move(n)) {}

    // The inferred result type of this expression.
    Type type() const;

    const Node& node() const { return *node_; }
    const std::shared_ptr<const Node>& ptr() const { return node_; }
    explicit operator bool() const { return static_cast<bool>(node_); }

   private:
    std::shared_ptr<const Node> node_;
};

enum class NodeKind { Column, Literal, Arith, Cmp, Logic, Cast };

// The concrete node. Public so the (internal) evaluator can switch on it; not
// part of the surface callers construct — use the builders below.
struct Node {
    NodeKind kind;
    Type type;  // inferred result type

    std::uint32_t col_index = 0;  // Column
    Scalar literal{};             // Literal
    ArithOp arith_op{};           // Arith
    CmpOp cmp_op{};               // Cmp
    LogicOp logic_op{};           // Logic
    Type cast_to = Type::I32;     // Cast (== type, kept for clarity)
    std::vector<Expr> children;   // 0 (leaf), 1 (NOT/Cast), or 2 (binary)
};

inline Type Expr::type() const { return node_->type; }

// ---- builders (type-inferring; throw std::invalid_argument on type error) --

// Column reference by physical index; caller states the column's type.
Expr col(Type type, std::uint32_t index);
// Column reference resolved by name against a Schema (to a physical index+type).
Expr col(const Schema& schema, std::string_view name);

// Literal.
Expr lit(Scalar s);

// Arithmetic (operands must be numeric: I32/I64/F64; mixed widths are promoted
// to a common type; the result has that promoted type).
Expr arith(ArithOp op, Expr a, Expr b);
Expr add(Expr a, Expr b);
Expr sub(Expr a, Expr b);
Expr mul(Expr a, Expr b);
Expr div(Expr a, Expr b);
Expr mod(Expr a, Expr b);

// Comparison (numeric operands are promoted to a common type; BOOL compares
// BOOL, TS compares TS, and — WP-7b — STR compares STR by resolved string
// VALUE (never by dictionary code); result is BOOL).
Expr cmp(CmpOp op, Expr a, Expr b);
Expr lt(Expr a, Expr b);
Expr le(Expr a, Expr b);
Expr gt(Expr a, Expr b);
Expr ge(Expr a, Expr b);
Expr eq(Expr a, Expr b);
Expr ne(Expr a, Expr b);

// Three-valued logical (operands must be BOOL; result is BOOL).
Expr logic_and(Expr a, Expr b);
Expr logic_or(Expr a, Expr b);
Expr logic_not(Expr a);

// Cast `e` to `to` (must be a meaningful conversion among the five Types).
Expr cast(Expr e, Type to);

// ---- evaluation ------------------------------------------------------------

// Which kernel family the evaluator drives. Vector is the production path;
// Scalar drives the independently-written reference twins and exists so tests
// can assert scalar == vector over whole expression trees. WP-3/WP-8 use the
// default (Vector) and need not mention this.
enum class Backend { Vector, Scalar };

// Evaluate `e` over `batch`, producing a dense OwnedColumn of batch.row_count
// values (the batch's selection vector is applied here; the result is dense and
// unselected). The result's Type is e.type().
OwnedColumn evaluate(const Expr& e, const Batch& batch,
                     Backend backend = Backend::Vector);

}  // namespace qe::expr
