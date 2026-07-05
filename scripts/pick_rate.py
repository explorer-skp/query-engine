#!/usr/bin/env python3
# ported from raft-rsm/bench/pick_rate.py — rigor infra, Raft-specifics stripped
"""Picks the headline offered load from a saved open-loop rate sweep.

Usage: pick_rate.py RESULTS_DIR PREFIX   (e.g. open.base)

Sustainable = the highest swept rate where every repeat stayed valid, abandoned
nothing, achieved >= 99% of the offered rate, AND kept the CO-corrected
intended-send tail bounded: p99 <= 20 ms and p99.9 <= 50 ms. (The two
thresholds were previously written in the swapped order — p99 <= 50 ms with
p99.9 <= 20 ms — which made the documented p99 clause dead code, since
p99.9 >= p99 always; audit fix.) The tail clauses matter: a bounded in-flight
generator can keep pace with the offered rate while per-request queueing grows
toward seconds, so rate fidelity alone does not mean "below the knee". The
headline load is 70% of that sustainable rate (a stated,
comfortably-inside-the-knee operating point). Falls back conservatively if
nothing qualifies. Prints one integer.

Reads the bench_skeleton JSON schema: config.rate, results.{valid,abandoned,
throughput_cps,e2e_intended.p99_ns,e2e_intended.p999_ns}.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    out, prefix = Path(sys.argv[1]), sys.argv[2]
    by_rate: dict[float, list[dict]] = {}
    for f in out.glob(f"{prefix}*.json"):
        d = json.loads(f.read_text())
        by_rate.setdefault(d["config"]["rate"], []).append(d["results"])

    sustainable = 0.0
    for rate in sorted(by_rate):
        runs = by_rate[rate]
        ok = all(
            r.get("valid", False)
            and r.get("abandoned", 1) == 0
            and r.get("throughput_cps", 0.0) >= 0.99 * rate
            and r.get("e2e_intended", {}).get("p99_ns", 1e18) <= 20e6
            and r.get("e2e_intended", {}).get("p999_ns", 1e18) <= 50e6
            for r in runs
        )
        if ok:
            sustainable = rate
    if sustainable <= 0:
        sustainable = min(by_rate) if by_rate else 2000.0
    print(int(sustainable * 0.7))


if __name__ == "__main__":
    main()
