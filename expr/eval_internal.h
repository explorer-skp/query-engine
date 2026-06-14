//  WP-2: INTERNAL eval helpers shared with the mutation self-test. Not a public
//  contract. The mutation test links a deliberately-broken variant of
//  propagate_nulls_and to prove the null-propagation check bites.
#pragma once

#include "core/column.h"
#include "core/owned_batch.h"

namespace qe::expr {

// Set `out`'s validity to (a valid AND b valid) — the "null-if-any" rule shared
// by arithmetic and comparison. `out` is a fresh length-n column (all_valid on
// entry); `a` and `b` are the two length-n operand views. Leaves `out` on the
// all_valid fast path iff both inputs are all-valid.
void propagate_nulls_and(OwnedColumn& out, const Column& a, const Column& b);

}  // namespace qe::expr
