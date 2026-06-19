//  WP-7b: StringDict — the out-of-band code->bytes table that backs a
//  dictionary-encoded VARCHAR column (Type::STR). A STR Column's physical `data`
//  holds int32 DICTIONARY CODES (byte_width(STR)==4); the StringDict resolves a
//  code to its string bytes. This is a HELPER TYPE (the frozen core/column.h
//  contract explicitly permits sibling helper types) — it is NOT part of the
//  frozen view surface; only the `const StringDict* dict` pointer on Column is.
//
//  CONTRACT:
//   * Interned + immutable-by-convention: build it up with intern() (which dedups
//     so equal strings share one code WITHIN this dict), then read it through
//     at()/size(). Equal strings in the SAME dict get the SAME code; the SAME
//     string in a DIFFERENT dict may get a DIFFERENT code. Therefore equality /
//     ordering of STR values is ALWAYS by resolved bytes (via at()), NEVER by raw
//     code — two columns may carry different dictionaries.
//   * From-scratch: a hand-rolled contiguous-bytes + offsets table plus a
//     std::unordered_map dedup index. No string/dataframe library (std:: only).
//   * No ISA / vector width / cache size / core count anywhere in its surface
//     (RIGOR.md interface discipline).
//
//  LAYOUT: bytes_ is all interned strings concatenated; offsets_ has size()+1
//  entries, string `code` occupies bytes_[offsets_[code] .. offsets_[code+1]).
//  Codes are assigned 0,1,2,... in first-seen order and never change.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qe {

class StringDict {
   public:
    StringDict() { offsets_.push_back(0); }

    // Intern `s`: if already present return its existing code; otherwise append
    // it and return a fresh code (== old size()). Dedups so equal strings share a
    // code within this dict.
    std::int32_t intern(std::string_view s) {
        std::string key(s);
        auto it = index_.find(key);
        if (it != index_.end()) return it->second;
        const auto code = static_cast<std::int32_t>(offsets_.size() - 1);
        bytes_.insert(bytes_.end(), s.begin(), s.end());
        offsets_.push_back(static_cast<std::uint32_t>(bytes_.size()));
        // The map owns its keys (std::string), so dedup never depends on bytes_'s
        // address (bytes_ may reallocate as it grows).
        index_.emplace(std::move(key), code);
        return code;
    }

    // The bytes of `code`. Precondition: 0 <= code < size().
    std::string_view at(std::int32_t code) const {
        const auto c = static_cast<std::size_t>(code);
        const std::uint32_t lo = offsets_[c];
        const std::uint32_t hi = offsets_[c + 1];
        return std::string_view(bytes_.data() + lo, hi - lo);
    }

    // Number of distinct interned strings (== next code to assign).
    std::size_t size() const noexcept { return offsets_.size() - 1; }

   private:
    std::vector<char> bytes_;          // all strings concatenated
    std::vector<std::uint32_t> offsets_;  // size()+1; string c = [off[c], off[c+1])
    std::unordered_map<std::string, std::int32_t> index_;  // value -> code (dedup)
};

}  // namespace qe
