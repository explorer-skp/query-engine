# expr/

Expression trees and their vectorized evaluation (predicates, arithmetic,
comparisons, three-valued logic, casts over column batches).

**WP-2 status:** implemented and frozen. `expr/expr.h` is the frozen public
contract (IR builders + `evaluate()`); WP-3/WP-8 build and evaluate trees through
it. See `WP-2-REPORT.md` for the API, the null-propagation truth table, and the
overflow policy.

Layout:
- `expr.h` / `expr.cpp` — frozen IR + type-inferring builders.
- `eval.cpp` — tree-walking evaluator; owns NULL propagation.
- `kernels.h` — internal op×type dispatch (the twin seam).
- `*_kernels.cpp` (vector, Highway) + `*_scalar.cpp` (independent scalar twins)
  for arith / compare / logic / cast.
- `expr_kernels_mutants.{h,cpp}` — test-only planted bugs for the mutation
  self-test.

From-scratch: no query-engine/dataframe libraries here; Highway only, behind the
`*_kernels.cpp` vector TUs.
