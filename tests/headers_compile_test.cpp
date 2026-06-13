// WP-0: header self-containment compile-check.
//
// Includes all four DRAFT interface headers and instantiates NOTHING. Its only
// job is to prove the headers compile and are self-contained (each pulls its own
// dependencies — e.g. column.h includes core/types.h, operator.h includes
// core/column.h). If a header forgets an include or a public type changes shape
// incompatibly, this translation unit fails to compile and ctest goes red.
//
// No behavior is exercised: WP-0 ships no engine logic.

#include "core/column.h"
#include "core/types.h"
#include "ops/operator.h"
#include "simd/kernel_convention.h"

// A few unevaluated type-level checks — no objects constructed, no logic run.
// These pin the DRAFT shapes so an accidental rename/retype is caught at compile
// time without instantiating anything.
static_assert(sizeof(qe::Type) > 0, "Type must be a complete enum");
static_assert(sizeof(qe::Column) > 0, "Column must be a complete type");
static_assert(sizeof(qe::Batch) > 0, "Batch must be a complete type");
static_assert(sizeof(qe::Schema) > 0, "Schema must be a complete type");
static_assert(sizeof(qe::SelectionVector) > 0, "SelectionVector must be complete");

int main() { return 0; }
