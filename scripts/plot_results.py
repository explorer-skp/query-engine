#!/usr/bin/env python3
# ported from raft-rsm/bench/plot_results.py — rigor infra, Raft-specifics stripped
"""Regenerates latency plots and a summary table from saved raw bench JSON.

Usage: plot_results.py RESULTS_DIR

Reads the per-run JSON files written by bench_skeleton, writes PNGs into
RESULTS_DIR/plots/ and a text summary into RESULTS_DIR/summary.txt. Never
re-runs anything: the saved files are the source of truth.

This is the domain-free reporting core: percentile axes, CDF-from-quantile-grid
curves, and the coordinated-omission ratio (intended tail / actual tail). It
makes no assumption about what the workload was — it reads whatever histograms
each run recorded under results.{e2e_intended,e2e_actual}.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# matplotlib is optional: the percentile/CDF PNGs need it, but the summary
# table is pure stdlib. Where matplotlib is absent we still emit the table and
# say so, rather than failing the whole report.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

US = 1e3  # ns -> us divisor


def load(outdir: Path):
    runs = defaultdict(list)  # label -> [doc...]
    for f in sorted(outdir.glob("*.json")):
        d = json.loads(f.read_text())
        runs[d.get("label", f.stem)].append(d)
    return runs


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def hist(doc, name):
    return doc.get("results", {}).get(name, {})


def quantile_curve(doc, name):
    """Returns (x_on_nines_axis, value_us) from a histogram's quantile grid."""
    qs = hist(doc, name).get("quantiles", [])
    xs = [1 / (1 - q) if q < 1 else 1e6 for q, _ in qs]
    ys = [v / US for _, v in qs]
    return xs, ys


def nines_axis(ax):
    ticks = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    ax.set_xscale("log")
    ax.set_xticks([1 / (1 - q) for q in ticks])
    ax.set_xticklabels(["p50", "p90", "p99", "p99.9", "p99.99", "p99.999"])
    ax.minorticks_off()


def plot_cdf(runs, plots: Path):
    """One percentile curve per (run, histogram): intended solid, actual dashed.

    When a stall is injected, the intended curve climbs into the tail while the
    actual curve stays flat — coordinated omission, shown."""
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False
    for label in sorted(runs):
        doc = runs[label][0]
        for name, ls in (("e2e_intended", "-"), ("e2e_actual", "--")):
            if not hist(doc, name).get("count"):
                continue
            xs, ys = quantile_curve(doc, name)
            if not xs:
                continue
            tag = name.replace("e2e_", "")
            ax.plot(xs, ys, ls, label=f"{label} {tag}")
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    nines_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("latency (us)")
    ax.set_title("Latency percentiles (intended vs actual send)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "latency_percentiles.png", dpi=130)
    plt.close(fig)


def write_summary(runs, outdir: Path):
    lines = []
    a = lines.append
    sample = next(iter(runs.values()))[0]
    machine = sample.get("machine_start", {})
    a("BENCHMARK SUMMARY (latencies in us; ratio = CO correction factor)")
    a(f"host: {machine.get('cpu_model', '?')} | "
      f"{machine.get('logical_cpus', '?')} cpus "
      f"(P={machine.get('perf_cores', '?')} E={machine.get('eff_cores', '?')}) | "
      f"isa={machine.get('isa', '?')} | {machine.get('kernel', '?')} | "
      f"git={machine.get('git_commit', '?')}")
    a("")
    head = (f"{'label':22s} {'thr/s':>9s} {'i-p50':>9s} {'i-p99':>9s} "
            f"{'i-p99.9':>9s} {'a-p99':>9s} {'ratio':>7s} {'valid':>6s}")
    a(head)
    a("-" * len(head))
    for label in sorted(runs):
        docs = runs[label]
        thr = med([d["results"].get("throughput_cps", float("nan"))
                   for d in docs])
        ip50 = med([hist(d, "e2e_intended").get("p50_ns", 0) for d in docs])
        ip99 = med([hist(d, "e2e_intended").get("p99_ns", 0) for d in docs])
        ip999 = med([hist(d, "e2e_intended").get("p999_ns", 0) for d in docs])
        ap99 = med([hist(d, "e2e_actual").get("p99_ns", 0) for d in docs])
        ratio = ip99 / ap99 if ap99 else float("nan")
        valid = all(d["results"].get("valid", False) for d in docs)
        a(f"{label:22s} {thr:9.0f} {ip50 / US:9.0f} {ip99 / US:9.0f} "
          f"{ip999 / US:9.0f} {ap99 / US:9.0f} {ratio:7.1f} {str(valid):>6s}")
    a("")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    outdir = Path(sys.argv[1])
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    runs = load(outdir)
    if not runs:
        print(f"no JSON results in {outdir}", file=sys.stderr)
        sys.exit(1)
    if HAVE_MPL:
        plot_cdf(runs, plots)
        print(f"plots in {plots}")
    else:
        print("matplotlib not installed: wrote summary.txt only, skipped PNGs",
              file=sys.stderr)
    write_summary(runs, outdir)


if __name__ == "__main__":
    main()
