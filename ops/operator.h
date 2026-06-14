//  FROZEN at WP-3. The universal pull-based operator contract — the substrate
//  every operator (this WP's scan/filter/project and every later WP's
//  aggregate/join/sort/asof/...) implements. Owned by the final review.
//
//  CONTRACT (frozen): the four pure-virtual methods below ARE the contract.
//  Later WPs may not add to, rename, or retype this base interface without an
//  Interface Change Request. Per-operator constructors and state are the
//  operator's own business; nothing operator-specific belongs here.
//
//  THE MODEL (decision D3 — pull-based, batch-at-a-time):
//   * open()  — prepare to produce (acquire child state, reset cursors). Call
//               exactly once before the first next(). An operator opens its
//               children in its own open() (it owns that wiring).
//   * next()  — produce ONE batch (~<=2048 logical rows) or std::nullopt to
//               signal exhaustion. The returned Batch is a VIEW (core/column.h);
//               its bytes are owned by the operator (typically an OwnedBatch
//               member) and remain valid ONLY until the next next()/close() call
//               on this operator. The caller must consume (or copy out of) the
//               batch before pulling again. A returned batch has row_count >= 1;
//               operators that would yield an empty batch instead pull again or
//               return nullopt. After the first nullopt, next() keeps returning
//               nullopt.
//   * close() — release resources; idempotent-friendly. After close(), the only
//               valid call is the destructor (or a fresh open()).
//   * output_schema() — the (name,Type) list of the columns next() yields; the
//               column order matches the Batch column order. Valid after
//               construction (does not require open()).
//
//  Width / ISA / cache / core-count never appear in this interface (Mac->x86
//  rule): a Batch is just typed columns; how wide the kernels run is the
//  kernels' private business.
#pragma once

#include <optional>

#include "core/column.h"

namespace qe {

class Operator {
   public:
    virtual void open() = 0;
    virtual std::optional<Batch> next() = 0;  // nullopt => exhausted
    virtual void close() = 0;
    virtual Schema output_schema() const = 0;
    virtual ~Operator() = default;
};

}  // namespace qe
