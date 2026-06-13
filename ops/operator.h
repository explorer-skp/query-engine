//  DRAFT — NOT FROZEN. Owned by the final review. Frozen at WP-1 (core/simd) /
//  WP-3 (Operator). Do not add fields, rename, or implement logic here.
//
//  WP-0 scaffold: non-binding signature stub. The pull-based operator contract
//  — open / next (one batch, nullopt => exhausted) / close — that composes the
//  vectorized execution pipeline. No operator is implemented in WP-0.
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
