//  WP-1 test seed accessor. The seed is chosen once in tests/wp1_test_main.cpp,
//  printed at startup, and overridable with `--seed N` for deterministic replay
//  (RIGOR.md rule 5). Every randomized WP-1 test derives its RNG from this.
#pragma once

#include <cstdint>

namespace qe::test {
std::uint64_t seed();
}  // namespace qe::test
