#!/usr/bin/env python3
# ported from raft-rsm/bench/plot_results.py — rigor infra, Raft-specifics stripped
# WP-10: extended to render the WP-10 doc families (vec-vs-scalar per operator,
# engine-vs-DuckDB ratio, roofline framing) ALONGSIDE the harvested coordinated-
# omission percentile/CDF view — all from SAVED JSON, never re-running the engine.
"""Regenerates latency plots and a summary table from saved raw bench JSON.

Usage: plot_results.py RESULTS_DIR

Reads the per-run JSON files written by the WP-10 bench drivers, writes PNGs into
RESULTS_DIR/plots/ and a text summary into RESULTS_DIR/summary.txt. Never re-runs
anything: the saved files are the source of truth (the reproducibility gate —
plots regenerate WITHOUT the engine).

Doc families, keyed by the top-level "kind":
  * bench_vec_scalar        — per-operator vector-vs-scalar relative ratio
                              (speedup scalar/vec at p50/p99), throughput
                              (rows/s, GB/s), and the roofline classification.
  * bench_engine_vs_duckdb  — engine-vs-DuckDB wall-clock ratio (one Plan, both
                              backends), when DuckDB was staged.
  * bench_sustained / bench_sort — coordinated-omission percentile/CDF view
                              (intended vs actual send time) + the CO ratio.

HONESTY (RIGOR.md §2): every caption echoes host+isa and, where the doc is Mac
(preliminary:true), prints "[PRELIMINARY / RELATIVE-ONLY]".
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

US = 1e3  # ns -> us divisor


def load(outdir: Path):
    docs = []
    for f in sorted(outdir.glob("*.json")):
        try:
            docs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)
    return docs


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def hist(doc, name):
    return doc.get("results", {}).get(name, {})


def by_kind(docs):
    groups = defaultdict(list)
    for d in docs:
        groups[d.get("kind", "unknown")].append(d)
    return groups


def prelim_tag(doc):
    return " [PRELIMINARY / RELATIVE-ONLY]" if doc.get("preliminary") else ""


def host_line(doc):
    m = doc.get("machine_start", {})
    return (f"host={doc.get('host', '?')} isa={doc.get('isa', '?')} | "
            f"{m.get('cpu_model', '?')} (P={m.get('perf_cores', '?')} "
            f"E={m.get('eff_cores', '?')}) | hwy={m.get('highway_target', '?')} "
            f"| git={m.get('git_commit', '?')}")


# ---- coordinated-omission percentile view (harvested; sustained + sort) -------

def quantile_curve(doc, name):
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


def plot_co_cdf(docs, plots: Path):
    """Intended (solid) vs actual (dashed) send-time percentiles. When a stall is
    injected the intended curve climbs into the tail while actual stays flat —
    coordinated omission, shown."""
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False
    for doc in docs:
        label = doc.get("label", "?")
        for name, ls in (("e2e_intended", "-"), ("e2e_actual", "--")):
            if not hist(doc, name).get("count"):
                continue
            xs, ys = quantile_curve(doc, name)
            if not xs:
                continue
            ax.plot(xs, ys, ls, label=f"{label} {name.replace('e2e_', '')}")
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    nines_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("latency (us)")
    ax.set_title("Coordinated-omission view: intended vs actual send")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "co_latency_percentiles.png", dpi=130)
    plt.close(fig)


def summarize_co(docs, a):
    a("COORDINATED-OMISSION (latencies in us; ratio = intended/actual tail)")
    head = (f"{'label':26s} {'thr/s':>9s} {'i-p50':>9s} {'i-p99':>9s} "
            f"{'i-p99.9':>9s} {'a-p99':>9s} {'CO×':>6s} {'valid':>6s}")
    a(head)
    a("-" * len(head))
    for doc in docs:
        ip50 = hist(doc, "e2e_intended").get("p50_ns", 0)
        ip99 = hist(doc, "e2e_intended").get("p99_ns", 0)
        ip999 = hist(doc, "e2e_intended").get("p999_ns", 0)
        ap99 = hist(doc, "e2e_actual").get("p99_ns", 0)
        ratio = ip99 / ap99 if ap99 else float("nan")
        thr = doc.get("results", {}).get("throughput_cps", float("nan"))
        valid = doc.get("results", {}).get("valid", False)
        a(f"{doc.get('label', '?'):26s} {thr:9.0f} {ip50 / US:9.0f} "
          f"{ip99 / US:9.0f} {ip999 / US:9.0f} {ap99 / US:9.0f} {ratio:6.1f} "
          f"{str(valid):>6s}{prelim_tag(doc)}")
    a("")


# ---- vec-vs-scalar ratio + roofline view (WP-10) -----------------------------

def plot_vec_scalar(docs, plots: Path):
    if not HAVE_MPL or not docs:
        return
    ops = [d.get("label", "?") for d in docs]
    sp50 = [d["results"].get("speedup_p50_scalar_over_vec", 0) for d in docs]
    sp99 = [d["results"].get("speedup_p99_scalar_over_vec", 0) for d in docs]
    x = range(len(ops))
    fig, ax = plt.subplots(figsize=(max(6, len(ops) * 1.4), 4.5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], sp50, w, label="p50")
    ax.bar([i + w / 2 for i in x], sp99, w, label="p99")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")  # 1.0 = vector == scalar
    ax.set_xticks(list(x))
    ax.set_xticklabels(ops, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("speedup  scalar / vector  (>1 = vector faster)")
    ax.set_title("Vec-vs-scalar per operator (relative ratio; >1 = vector wins)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "vec_vs_scalar_speedup.png", dpi=130)
    plt.close(fig)


def summarize_vec_scalar(docs, a):
    a("VEC-VS-SCALAR per operator (speedup = scalar/vector; >1 => vector faster)")
    head = (f"{'operator':18s} {'rows':>9s} {'vec p50us':>10s} "
            f"{'sca p50us':>10s} {'sp p50':>7s} {'sp p99':>7s} "
            f"{'vec GB/s':>9s} {'AI f/B':>7s} {'roofline':>15s} {'valid':>6s}")
    a(head)
    a("-" * len(head))
    for doc in docs:
        r = doc.get("results", {})
        vp = hist(doc, "vec_ns").get("p50_ns", 0) / US
        sp = hist(doc, "scalar_ns").get("p50_ns", 0) / US
        thr = r.get("throughput", {})
        rl = r.get("roofline", {})
        a(f"{doc.get('operator', '?'):18s} "
          f"{doc.get('config', {}).get('rows', 0):9d} {vp:10.1f} {sp:10.1f} "
          f"{r.get('speedup_p50_scalar_over_vec', 0):7.2f} "
          f"{r.get('speedup_p99_scalar_over_vec', 0):7.2f} "
          f"{thr.get('vec_gb_per_s', 0):9.2f} "
          f"{rl.get('arithmetic_intensity_flops_per_byte', 0):7.3f} "
          f"{rl.get('classification', '?'):>15s} "
          f"{str(r.get('valid', False)):>6s}")
    a(f"  ({prelim_tag(docs[0]).strip() or 'host-governed'}; roofline ridge "
      f"ASSUMED on Mac — credible roofline is x86, §2)")
    a("")


def summarize_sort_gather(docs, a):
    """The WP-7 permutation-gather microbench (its own schema): vec/scalar gather
    histograms + speedup. Reported here so its saved JSON is not orphaned."""
    a("PERMUTATION-GATHER microbench (WP-7 worked example; speedup scalar/vector)")
    head = (f"{'label':14s} {'elems':>9s} {'vec p50us':>10s} {'sca p50us':>10s} "
            f"{'sp p50':>7s} {'sp p99':>7s}")
    a(head)
    a("-" * len(head))
    for doc in docs:
        r = doc.get("results", {})
        vp = r.get("gather64_vec_ns", {}).get("p50_ns", 0) / US
        sp = r.get("gather64_scalar_ns", {}).get("p50_ns", 0) / US
        a(f"{doc.get('label', '?'):14s} "
          f"{doc.get('config', {}).get('elements', 0):9d} {vp:10.1f} {sp:10.1f} "
          f"{r.get('speedup_p50_scalar_over_vec', 0):7.2f} "
          f"{r.get('speedup_p99_scalar_over_vec', 0):7.2f}{prelim_tag(doc)}")
    a("")


def summarize_engine_vs_duckdb(docs, a):
    a("ENGINE-VS-DUCKDB (wall-clock; ratio = duckdb/engine at p50; >1 => engine "
      "faster)")
    head = (f"{'label':20s} {'rows':>9s} {'eng p50us':>10s} "
            f"{'duck p50us':>11s} {'duck/eng':>9s} {'staged':>7s} {'valid':>6s}")
    a(head)
    a("-" * len(head))
    for doc in docs:
        r = doc.get("results", {})
        ep = hist(doc, "engine_ns").get("p50_ns", 0) / US
        dp = hist(doc, "duckdb_ns").get("p50_ns", 0) / US
        a(f"{doc.get('label', '?'):20s} "
          f"{doc.get('config', {}).get('rows', 0):9d} {ep:10.1f} {dp:11.1f} "
          f"{r.get('speedup_p50_duckdb_over_engine', 0):9.2f} "
          f"{str(r.get('duckdb_available', False)):>7s} "
          f"{str(r.get('valid', False)):>6s}{prelim_tag(doc)}")
    a("  (note: DuckDB time includes per-call table ingest — relative indicator, "
      "§2)")
    a("")


def main() -> None:
    outdir = Path(sys.argv[1])
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    docs = load(outdir)
    if not docs:
        print(f"no JSON results in {outdir}", file=sys.stderr)
        sys.exit(1)
    groups = by_kind(docs)

    lines = []
    a = lines.append
    a("BENCHMARK SUMMARY — regenerated from saved JSON (engine NOT re-run)")
    a(host_line(docs[0]))
    a("")

    if groups.get("bench_vec_scalar"):
        summarize_vec_scalar(groups["bench_vec_scalar"], a)
        plot_vec_scalar(groups["bench_vec_scalar"], plots)
    if groups.get("bench_engine_vs_duckdb"):
        summarize_engine_vs_duckdb(groups["bench_engine_vs_duckdb"], a)
    if groups.get("bench_sort"):
        summarize_sort_gather(groups["bench_sort"], a)
    co_docs = groups.get("bench_sustained", [])
    if co_docs:
        summarize_co(co_docs, a)
        plot_co_cdf(co_docs, plots)

    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if HAVE_MPL:
        print(f"plots in {plots}")
    else:
        print("matplotlib not installed: wrote summary.txt only, skipped PNGs",
              file=sys.stderr)


if __name__ == "__main__":
    main()
