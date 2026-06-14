//  WP-4 FUZZ: cross-check the real HashTable against an INDEPENDENT
//  std::unordered_map reference model on randomized insert/find sequences. The
//  table must agree with the model on EVERY group id and EVERY membership answer.
//  This pins the §12 hotspot — hash-table probe overflow / capacity growth — and
//  is the differential the mutation self-test (hashtable_mutation_test.cpp) shows
//  can bite. Seeded; replay any failure with:  ./hashtable_fuzz_test --seed N
//
//  The reference keys rows by the SAME canonical key the engine computes
//  (null-mask + normalized words), via the shared normalization — so this test
//  targets the TABLE MECHANICS (probe walk, growth/rehash, membership), while
//  normalization itself (-0.0/NaN, sign/zero-extension) is pinned by the unit
//  tests. Group ids stay in lockstep because both assign ids 0,1,2,... in
//  first-seen row order (including kNeverMatch "dead" null groups).

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "doctest/doctest.h"

#include "core/types.h"
#include "ops/hash_internal.h"
#include "ops/hashtable.h"
#include "tests/hashtable_test_util.h"
#include "tests/wp1_seed.h"

using namespace qe;
using namespace qe::ht_test;

namespace {

// Canonical key of a row = its null mask followed by the normalized word of each
// NON-null column (null columns contribute only the mask bit). Mirrors the
// engine's kEqual equality exactly.
struct RowKey {
    std::uint64_t null_mask;
    std::vector<std::uint64_t> words;  // only the non-null columns, in order
    bool operator==(const RowKey& o) const {
        return null_mask == o.null_mask && words == o.words;
    }
};
struct RowKeyHash {
    std::size_t operator()(const RowKey& k) const {
        std::size_t h = std::hash<std::uint64_t>{}(k.null_mask);
        for (std::uint64_t w : k.words)
            h ^= std::hash<std::uint64_t>{}(w) + 0x9e3779b97f4a7c15ull +
                 (h << 6) + (h >> 2);
        return h;
    }
};

// Independent reference model honoring the chosen NullPolicy.
class RefModel {
   public:
    RefModel(NullPolicy pol, std::size_t ncols) : pol_(pol), ncols_(ncols) {}

    std::uint32_t insert_or_find(std::uint64_t null_mask,
                                 const std::vector<std::uint64_t>& words_at) {
        if (pol_ == NullPolicy::kNeverMatch && null_mask != 0) {
            return next_++;  // dead, unique, never deduped
        }
        const RowKey key = make_key(null_mask, words_at);
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        const std::uint32_t id = next_++;
        map_.emplace(key, id);
        return id;
    }

    std::uint32_t find(std::uint64_t null_mask,
                       const std::vector<std::uint64_t>& words_at) const {
        if (pol_ == NullPolicy::kNeverMatch && null_mask != 0) return kNoGroup;
        const RowKey key = make_key(null_mask, words_at);
        auto it = map_.find(key);
        return it == map_.end() ? kNoGroup : it->second;
    }

    std::uint32_t num_groups() const { return next_; }

   private:
    RowKey make_key(std::uint64_t null_mask,
                    const std::vector<std::uint64_t>& words_at) const {
        RowKey k;
        k.null_mask = null_mask;
        for (std::size_t j = 0; j < ncols_; ++j)
            if (!((null_mask >> j) & 1ull)) k.words.push_back(words_at[j]);
        return k;
    }
    NullPolicy pol_;
    std::size_t ncols_;
    std::unordered_map<RowKey, std::uint32_t, RowKeyHash> map_;
    std::uint32_t next_ = 0;
};

// Compute the engine-canonical (null_mask, per-column words) for a batch so the
// reference sees exactly the engine's keys.
void canonicalize(const KeyColumns& kc, std::size_t n,
                  const std::vector<Type>& types,
                  std::vector<std::uint64_t>& null_mask,
                  std::vector<std::vector<std::uint64_t>>& words) {
    std::vector<std::uint64_t> hash;
    ops::detail::normalize_and_hash(kc, n, types, kDefaultHashSeed,
                                    HashPath::kVector, words, null_mask, hash);
}

}  // namespace

TEST_CASE("fuzz: random insert/find mixes agree with the reference model") {
    std::mt19937_64 rng(qe::test::seed() ^ 0xF022u);
    for (int trial = 0; trial < 60; ++trial) {
        const std::size_t ncols = 1 + rng() % 3;
        const NullPolicy pol =
            (rng() & 1) ? NullPolicy::kEqual : NullPolicy::kNeverMatch;
        std::vector<Type> types;
        for (std::size_t j = 0; j < ncols; ++j) {
            switch (rng() % 5) {
                case 0: types.push_back(Type::I32); break;
                case 1: types.push_back(Type::I64); break;
                case 2: types.push_back(Type::F64); break;
                case 3: types.push_back(Type::BOOL); break;
                default: types.push_back(Type::TS); break;
            }
        }
        // Small initial capacity so growth fires often.
        HashTableConfig cfg;
        cfg.initial_capacity = 8;
        HashTable t(types, pol, cfg, rng());
        RefModel ref(pol, ncols);

        const int domain = 1 + static_cast<int>(rng() % 40);  // collision pressure
        for (int round = 0; round < 8; ++round) {
            const std::size_t n = rng() % 300;
            const bool do_insert = (rng() % 3) != 0;  // mostly inserts

            // Build n rows over a small domain (forces dedup + collisions).
            std::vector<std::vector<std::int64_t>> cols(ncols);
            std::vector<std::vector<std::size_t>> nulls(ncols);
            for (std::size_t j = 0; j < ncols; ++j) {
                cols[j].resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    if (rng() % 7 == 0) {  // ~14% nulls
                        cols[j][i] = 0;
                        nulls[j].push_back(i);
                    } else if (types[j] == Type::F64) {
                        const int pick = static_cast<int>(rng() % domain);
                        cols[j][i] = f64_bits(static_cast<double>(pick) * 0.5);
                    } else {
                        cols[j][i] = static_cast<std::int64_t>(rng() % domain);
                    }
                }
            }
            KeyHolder h;
            for (std::size_t j = 0; j < ncols; ++j)
                h.owned.push_back(make_col(types[j], cols[j], nulls[j]));
            const KeyColumns kc = h.kc();

            std::vector<std::uint64_t> nm;
            std::vector<std::vector<std::uint64_t>> wds;
            canonicalize(kc, n, types, nm, wds);

            std::vector<std::uint32_t> got(n);
            if (do_insert)
                t.insert_or_find(kc, n, got.data());
            else
                t.find(kc, n, got.data());

            for (std::size_t i = 0; i < n; ++i) {
                std::vector<std::uint64_t> wi(ncols);
                for (std::size_t j = 0; j < ncols; ++j) wi[j] = wds[j][i];
                const std::uint32_t want =
                    do_insert ? ref.insert_or_find(nm[i], wi)
                              : ref.find(nm[i], wi);
                REQUIRE(got[i] == want);
            }
            REQUIRE(t.num_groups() == ref.num_groups());
        }
    }
}

TEST_CASE("fuzz: collision storm — many keys mapped to one bucket chain") {
    // Distinct keys whose hashes are spread by the mixer, but we force a small
    // table so a long probe chain forms; correctness must hold and find() must
    // recover every key from anywhere in the chain.
    std::mt19937_64 rng(qe::test::seed() ^ 0xC011u);
    HashTableConfig cfg;
    cfg.initial_capacity = 8;       // tiny: forces long chains + many growths
    cfg.max_load_factor = 0.9;      // pack densely
    HashTable t({Type::I64}, NullPolicy::kEqual, cfg, rng());
    RefModel ref(NullPolicy::kEqual, 1);

    const std::size_t N = 5000;
    std::vector<std::int64_t> keys(N);
    for (std::size_t i = 0; i < N; ++i) keys[i] = static_cast<std::int64_t>(i);

    KeyHolder h;
    h.owned.push_back(make_col(Type::I64, keys));
    const KeyColumns kc = h.kc();
    std::vector<std::uint64_t> nm;
    std::vector<std::vector<std::uint64_t>> wds;
    canonicalize(kc, N, {Type::I64}, nm, wds);

    std::vector<std::uint32_t> got(N);
    t.insert_or_find(kc, N, got.data());
    for (std::size_t i = 0; i < N; ++i) {
        std::vector<std::uint64_t> wi{wds[0][i]};
        REQUIRE(got[i] == ref.insert_or_find(nm[i], wi));
    }
    REQUIRE(t.num_groups() == N);  // all distinct

    // Every key is findable after all the growths.
    std::vector<std::uint32_t> fg(N);
    t.find(kc, N, fg.data());
    for (std::size_t i = 0; i < N; ++i) REQUIRE(fg[i] == got[i]);
}

TEST_CASE("fuzz: insert past every resize boundary keeps all keys") {
    // Insert a long run of distinct keys one batch at a time from a tiny initial
    // capacity, crossing many doublings; after each batch, EVERY key inserted so
    // far must still be findable (catches a botched rehash on growth).
    std::mt19937_64 rng(qe::test::seed() ^ 0x5121u);
    HashTableConfig cfg;
    cfg.initial_capacity = 8;
    HashTable t({Type::I32}, NullPolicy::kEqual, cfg, rng());

    std::vector<std::int32_t> all;
    for (int batch = 0; batch < 30; ++batch) {
        std::vector<std::int64_t> add;
        const std::size_t m = 1 + rng() % 50;
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t v = static_cast<std::int32_t>(all.size());
            all.push_back(v);
            add.push_back(v);
        }
        KeyHolder h;
        h.owned.push_back(make_col(Type::I32, add));
        std::vector<std::uint32_t> g(m);
        t.insert_or_find(h.kc(), m, g.data());

        // All keys 0..all.size()-1 are present with id == value (dense, in order).
        std::vector<std::int64_t> probe(all.begin(), all.end());
        KeyHolder ph;
        ph.owned.push_back(make_col(Type::I32, probe));
        std::vector<std::uint32_t> fg(all.size());
        t.find(ph.kc(), all.size(), fg.data());
        for (std::size_t i = 0; i < all.size(); ++i) REQUIRE(fg[i] == i);
    }
    REQUIRE(t.num_groups() == all.size());
}
