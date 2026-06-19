//  WP-7b mutation library — the planted STRING-KEY defect, kept in its OWN
//  DISTINCTLY-NAMED enum (qe::mutant::StringKeyMutation) so it never collides with
//  the join/sort/aggregate/asof/window Mutation enums (the conflict-free-TU
//  convention). TEST-ONLY: never linked into qe_ops / the engine.
//
//  THE DEFECT (rule 4 / the §5 "by value, not by code" hazard). A correct equi-join
//  on a dictionary-encoded STR key must compare the string VALUES: the build side
//  and the probe side carry INDEPENDENT dictionaries, so the SAME string usually
//  gets DIFFERENT int32 codes on the two sides. The real HashJoin canonicalizes
//  both sides' STR keys into one shared VALUE-id space before the hash table
//  (ops/join.cpp). The mutant below is a faithful copy that SKIPS that step and
//  feeds the RAW codes to the table — so two equal strings with different codes
//  miss the join. The DuckDB differential (real VARCHAR join) catches it.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/column.h"
#include "ops/join.h"  // JoinType, GatherPath
#include "ops/operator.h"

namespace qe::mutant {

enum class StringKeyMutation {
    kNone,        // faithful real behavior (control)
    kHashRawCode  // hash/compare STR keys by raw int32 CODE, never by value
};

// A hash equi-join that mirrors qe::HashJoin EXACTLY except for the STR key
// handling selected by `mutation`. With kNone it canonicalizes STR keys by value
// (== the real operator); with kHashRawCode it feeds raw STR codes to the table.
class StringKeyJoin : public Operator {
   public:
    StringKeyJoin(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
                  std::vector<std::uint32_t> probe_keys,
                  std::vector<std::uint32_t> build_keys, qe::JoinType type,
                  StringKeyMutation mutation);

    void open() override;
    std::optional<Batch> next() override;
    void close() override;
    Schema output_schema() const override;

   private:
    static constexpr std::size_t kOutBatch = 2048;
    void build_side();
    bool build_pairs_for_probe();
    void make_key_views(const std::vector<std::uint32_t>& keys, const Batch& b);

    std::unique_ptr<Operator> probe_;
    std::unique_ptr<Operator> build_;
    std::vector<std::uint32_t> probe_keys_;
    std::vector<std::uint32_t> build_keys_;
    qe::JoinType type_;
    StringKeyMutation mutation_;
    Schema probe_schema_;
    Schema build_schema_;

    struct State;
    std::shared_ptr<State> state_;
    std::optional<Batch> cur_probe_;
    std::size_t pair_cursor_ = 0;
    bool opened_ = false;
    bool probe_done_ = false;
    OwnedBatch current_;
};

}  // namespace qe::mutant
