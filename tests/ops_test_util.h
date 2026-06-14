//  WP-3 test helpers: build small Tables from literals, and a one-shot source
//  operator that can emit a batch carrying a selection vector (to exercise
//  Filter/Project on an already-selected input — §12 / the WP-3 "Done" list).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/column.h"
#include "core/owned_batch.h"
#include "core/types.h"
#include "ops/operator.h"
#include "ops/table.h"

namespace qe::ops_test {

// An I32 column from values; indices in `nulls` are marked NULL.
inline OwnedColumn i32_col(const std::vector<std::int32_t>& vals,
                           const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::I32, vals.size());
    auto* d = reinterpret_cast<std::int32_t*>(c.mutable_data());
    for (std::size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

inline OwnedColumn f64_col(const std::vector<double>& vals,
                           const std::vector<std::size_t>& nulls = {}) {
    OwnedColumn c = OwnedColumn::make(Type::F64, vals.size());
    auto* d = reinterpret_cast<double*>(c.mutable_data());
    for (std::size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
    for (std::size_t k : nulls) c.set_null(k);
    return c;
}

// A one-batch source: emits the given (externally owned) OwnedBatch's view once,
// then nullopt. Lets a test feed Filter/Project a batch with a selection vector.
class OneBatchSource : public Operator {
   public:
    OneBatchSource(Schema schema, const OwnedBatch& batch)
        : schema_(std::move(schema)), batch_(&batch) {}

    void open() override { done_ = false; }
    std::optional<Batch> next() override {
        if (done_) return std::nullopt;
        done_ = true;
        return batch_->view();
    }
    void close() override {}
    Schema output_schema() const override { return schema_; }

   private:
    Schema schema_;
    const OwnedBatch* batch_;
    bool done_ = false;
};

}  // namespace qe::ops_test
