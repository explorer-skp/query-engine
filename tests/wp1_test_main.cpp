//  WP-1 test entry point. Like tests/test_main.cpp it provides doctest's main,
//  but it first extracts an optional `--seed N` (printing the seed used, random
//  if absent) so every randomized WP-1 test replays deterministically:
//      ./validity_bitmap_test --seed 12345
//  The remaining args are forwarded to doctest unchanged.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "tests/wp1_seed.h"

namespace qe::test {
namespace {
std::uint64_t g_seed = 0;
}
std::uint64_t seed() { return g_seed; }
}  // namespace qe::test

int main(int argc, char** argv) {
    bool have_seed = false;
    std::vector<char*> passthru;
    passthru.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            qe::test::g_seed = std::strtoull(argv[i + 1], nullptr, 10);
            have_seed = true;
            ++i;  // consume the value
        } else {
            passthru.push_back(argv[i]);
        }
    }
    if (!have_seed) {
        std::random_device rd;
        qe::test::g_seed =
            (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    std::printf("[wp1] seed=%llu  (replay this run with: %s --seed %llu)\n",
                static_cast<unsigned long long>(qe::test::g_seed), argv[0],
                static_cast<unsigned long long>(qe::test::g_seed));
    std::fflush(stdout);

    doctest::Context ctx;
    ctx.applyCommandLine(static_cast<int>(passthru.size()), passthru.data());
    return ctx.run();
}
