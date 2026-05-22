#!/usr/bin/env python3
"""Post-process experiment CSVs into publication-quality figures and tables.

Reads every ``*_results.csv`` under ``experiments/results/``, aggregates per
configuration, and writes:

* PNG figures to ``figures/`` (one per analysis).
* Markdown tables to ``doc/tables/`` (drop-in for the memoria).
* A LaTeX-ready summary in ``doc/tables/summary.tex``.

Robust to partial data: skips configs with zero rows so it can be re-run while
the long experiment is still in flight. Re-runs are idempotent.

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Lazy matplotlib import so a `--help` invocation doesn't pull it in
_RESULTS_DIR_DEFAULT = Path("experiments/results")
_FIGURES_DIR_DEFAULT = Path("figures")
_TABLES_DIR_DEFAULT = Path("doc/tables")

# Display order and human labels — keeps figures and tables consistent.
_CONFIG_ORDER = [
    "baseline",
    "sequential",
    "self_reflection_r1",
    "self_reflection_r2",
    "self_reflection_r3",
    "ablation_no_pm",
    "ablation_no_architect",
    "ablation_no_reviewer",
]
_CONFIG_LABELS = {
    "baseline": "Baseline",
    "sequential": "Sequential",
    "self_reflection_r1": "SR (r=1)",
    "self_reflection_r2": "SR (r=2)",
    "self_reflection_r3": "SR (r=3)",
    "ablation_no_pm": "Abl. -PM",
    "ablation_no_architect": "Abl. -Arch.",
    "ablation_no_reviewer": "Abl. -Rev.",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Parse one CSV row into typed values, dropping malformed entries."""
    try:
        return {
            "benchmark": row["benchmark"],
            "problem_id": row["problem_id"],
            "config": row["config"],
            "seed": int(row["seed"]),
            "pass_all_tests": row["pass_all_tests"].strip().lower() == "true",
            "test_pass_rate": float(row["test_pass_rate"] or 0.0),
            "tokens_input": int(row["tokens_input"] or 0),
            "tokens_output": int(row["tokens_output"] or 0),
            "latency_seconds": float(row["latency_seconds"] or 0.0),
            "revision_count": int(row["revision_count"] or 0),
            "error": row.get("error", "").strip(),
        }
    except (KeyError, ValueError):
        return None


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every row from every ``*_results.csv`` (excluding quick_check)."""
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        if csv_path.name == "quick_check.csv":
            continue
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                parsed = _coerce_row(raw)
                if parsed is None:
                    continue
                rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    log_num = math.lgamma(n - c + 1) - math.lgamma(n - c - k + 1)
    log_den = math.lgamma(n + 1) - math.lgamma(n - k + 1)
    return 1.0 - math.exp(log_num - log_den)


def bootstrap_pass_at_1_ci(
    passes: list[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for pass@1.

    Returns (low, high) at the (1-alpha) confidence level. Uses a fixed
    seed so the CI is stable across re-runs of the analysis.
    """
    if not passes:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(passes)
    samples: list[float] = []
    for _ in range(n_boot):
        draws = sum(passes[rng.randrange(n)] for _ in range(n))
        samples.append(draws / n)
    samples.sort()
    lo = samples[int((alpha / 2) * n_boot)]
    hi = samples[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def aggregate_by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute summary metrics per config."""
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        by_config[r["config"]].append(r)

    summary: dict[str, dict[str, float]] = {}
    for cfg, group in by_config.items():
        n_runs = len(group)
        n_pass = sum(1 for r in group if r["pass_all_tests"])

        # pass@1 across the full sample, then per-problem pass@k for k=3 (3 seeds).
        pass_at_1 = n_pass / n_runs if n_runs else 0.0

        per_problem: dict[str, list[bool]] = defaultdict(list)
        for r in group:
            per_problem[r["problem_id"]].append(r["pass_all_tests"])
        k = max(3, 1)
        pass_at_k_vals = []
        for pid, results in per_problem.items():
            n = len(results)
            c = sum(results)
            pass_at_k_vals.append(_pass_at_k(n, c, min(k, n)))
        pass_at_3 = (
            sum(pass_at_k_vals) / len(pass_at_k_vals) if pass_at_k_vals else 0.0
        )

        toks_in = [r["tokens_input"] for r in group]
        toks_out = [r["tokens_output"] for r in group]
        lat = [r["latency_seconds"] for r in group]
        revs = [r["revision_count"] for r in group]

        ci_lo, ci_hi = bootstrap_pass_at_1_ci(
            [r["pass_all_tests"] for r in group]
        )
        summary[cfg] = {
            "n_runs": float(n_runs),
            "n_problems": float(len(per_problem)),
            "pass_at_1": pass_at_1,
            "pass_at_1_ci_low": ci_lo,
            "pass_at_1_ci_high": ci_hi,
            "pass_at_3": pass_at_3,
            "mean_tokens_in": statistics.fmean(toks_in) if toks_in else 0.0,
            "mean_tokens_out": statistics.fmean(toks_out) if toks_out else 0.0,
            "mean_tokens_total": (
                statistics.fmean(t1 + t2 for t1, t2 in zip(toks_in, toks_out))
                if toks_in else 0.0
            ),
            "mean_latency_s": statistics.fmean(lat) if lat else 0.0,
            "median_latency_s": statistics.median(lat) if lat else 0.0,
            "mean_revisions": statistics.fmean(revs) if revs else 0.0,
        }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ordered_configs(present: set[str]) -> list[str]:
    return [c for c in _CONFIG_ORDER if c in present] + sorted(
        present - set(_CONFIG_ORDER)
    )


def plot_pass_at_1(
    summary: dict[str, dict[str, float]], out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    cfgs = _ordered_configs(set(summary))
    values = [summary[c]["pass_at_1"] for c in cfgs]
    lows = [summary[c].get("pass_at_1_ci_low", values[i]) for i, c in enumerate(cfgs)]
    highs = [summary[c].get("pass_at_1_ci_high", values[i]) for i, c in enumerate(cfgs)]
    errs_low = [v - l for v, l in zip(values, lows)]
    errs_high = [h - v for v, h in zip(values, highs)]
    labels = [_CONFIG_LABELS.get(c, c) for c in cfgs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        labels,
        values,
        edgecolor="black",
        yerr=[errs_low, errs_high],
        capsize=4,
    )
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, max(values + [0.0]) * 1.15 if values else 1.0)
    ax.set_title("pass@1 por configuración")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cost_quality(
    summary: dict[str, dict[str, float]], out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    cfgs = _ordered_configs(set(summary))
    xs = [summary[c]["mean_tokens_total"] for c in cfgs]
    ys = [summary[c]["pass_at_1"] for c in cfgs]
    labels = [_CONFIG_LABELS.get(c, c) for c in cfgs]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=80, edgecolor="black")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            lbl,
            (x, y),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Tokens medios por problema (escala log)")
    ax.set_ylabel("pass@1")
    ax.set_title("Coste vs. calidad — frontera de Pareto")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_latency_box(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    by_config: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        by_config[r["config"]].append(r["latency_seconds"])

    cfgs = _ordered_configs(set(by_config))
    data = [by_config[c] for c in cfgs]
    labels = [_CONFIG_LABELS.get(c, c) for c in cfgs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Latencia por problema (s)")
    ax.set_title("Distribución de latencia por configuración")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_revision_distribution(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    """Stacked bar showing how often each revision count is used in SR configs."""
    import matplotlib.pyplot as plt

    targets = [c for c in _CONFIG_ORDER if c.startswith("self_reflection")]
    present = {r["config"] for r in rows if not r["error"]}
    targets = [c for c in targets if c in present]
    if not targets:
        return

    counts: dict[str, dict[int, int]] = {c: defaultdict(int) for c in targets}
    for r in rows:
        if r["error"] or r["config"] not in counts:
            continue
        counts[r["config"]][r["revision_count"]] += 1

    max_rev = max(
        (max(d.keys()) if d else 0 for d in counts.values()), default=0
    )
    levels = list(range(max_rev + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = [0] * len(targets)
    for lvl in levels:
        vals = [counts[c].get(lvl, 0) for c in targets]
        ax.bar(
            [_CONFIG_LABELS[c] for c in targets],
            vals,
            bottom=bottoms,
            label=f"r={lvl}",
            edgecolor="black",
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Número de runs")
    ax.set_title("Distribución de iteraciones de revisión")
    ax.legend(title="revision_count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_summary_markdown(
    summary: dict[str, dict[str, float]], out_path: Path
) -> None:
    cfgs = _ordered_configs(set(summary))
    header = (
        "| Configuración | n_runs | pass@1 (IC 95%) | pass@3 | "
        "Tokens medios | Latencia media (s) | Revisiones medias |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for c in cfgs:
        s = summary[c]
        lo = s.get("pass_at_1_ci_low", s["pass_at_1"])
        hi = s.get("pass_at_1_ci_high", s["pass_at_1"])
        lines.append(
            f"| {_CONFIG_LABELS.get(c, c)} | "
            f"{int(s['n_runs'])} | "
            f"{s['pass_at_1']:.2%} [{lo:.2%}, {hi:.2%}] | "
            f"{s['pass_at_3']:.2%} | "
            f"{s['mean_tokens_total']:.0f} | "
            f"{s['mean_latency_s']:.2f} | "
            f"{s['mean_revisions']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_latex(
    summary: dict[str, dict[str, float]], out_path: Path
) -> None:
    cfgs = _ordered_configs(set(summary))
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Configuración & n & pass@1 & pass@3 & Tokens & Latencia (s) & Revisiones \\\\",
        "\\midrule",
    ]
    for c in cfgs:
        s = summary[c]
        lines.append(
            f"{_CONFIG_LABELS.get(c, c)} & "
            f"{int(s['n_runs'])} & "
            f"{s['pass_at_1']*100:.2f}\\% & "
            f"{s['pass_at_3']*100:.2f}\\% & "
            f"{s['mean_tokens_total']:.0f} & "
            f"{s['mean_latency_s']:.2f} & "
            f"{s['mean_revisions']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_difficulty_report(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    """Per-problem difficulty across configs.

    Reports the problems where the spread of pass rates between configs
    is largest — those are the most informative for the discussion section.
    """
    per: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        per[(r["benchmark"], r["problem_id"], r["config"])].append(
            r["pass_all_tests"]
        )

    # Build per-problem rows: {(bm, pid): {cfg: pass_rate}}
    by_problem: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (bm, pid, cfg), results in per.items():
        by_problem[(bm, pid)][cfg] = sum(results) / len(results)

    present_configs = _ordered_configs(
        {cfg for cfgs in by_problem.values() for cfg in cfgs}
    )
    if len(present_configs) < 2:
        return

    # Compute spread = max - min across configs that ran the problem.
    scored: list[tuple[float, tuple[str, str], dict[str, float]]] = []
    for key, cfg_rates in by_problem.items():
        vals = list(cfg_rates.values())
        spread = max(vals) - min(vals)
        scored.append((spread, key, cfg_rates))

    # Top-20 hardest-to-split problems (largest spread first)
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:20]

    header = (
        "| Benchmark | Problema | "
        + " | ".join(_CONFIG_LABELS.get(c, c) for c in present_configs)
        + " | Spread |"
    )
    sep = "|---|---|" + "---:|" * (len(present_configs) + 1)
    lines = [
        "## Problemas con mayor desacuerdo entre configuraciones (top 20)",
        "",
        header,
        sep,
    ]
    for spread, (bm, pid), cfg_rates in top:
        cells = []
        for c in present_configs:
            if c in cfg_rates:
                cells.append(f"{cfg_rates[c]:.0%}")
            else:
                cells.append("—")
        lines.append(
            f"| {bm} | {pid} | " + " | ".join(cells) + f" | {spread:.0%} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_benchmark_table(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    """Cross-tab: config × benchmark → pass@1."""
    bm: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        bm[(r["config"], r["benchmark"])].append(r["pass_all_tests"])

    cfgs = _ordered_configs({c for c, _ in bm})
    benchmarks = sorted({b for _, b in bm})
    if not benchmarks:
        return

    header = "| Configuración | " + " | ".join(benchmarks) + " |"
    sep = "|---|" + "---:|" * len(benchmarks)
    lines = [header, sep]
    for c in cfgs:
        cells = []
        for b in benchmarks:
            results = bm.get((c, b), [])
            if not results:
                cells.append("—")
            else:
                cells.append(f"{sum(results) / len(results):.2%}")
        lines.append(f"| {_CONFIG_LABELS.get(c, c)} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

@dataclass
class McNemarResult:
    """Outcome of a paired McNemar test for two configurations."""

    b: int
    c: int
    n_pairs: int
    statistic: float
    p_value: float
    method: str


def mcnemar_test(
    config_a: list[tuple[str, int, bool]],
    config_b: list[tuple[str, int, bool]],
) -> McNemarResult | None:
    """Paired McNemar test on matched (problem_id, seed) outcomes.

    Uses the chi-square statistic with continuity correction when the
    number of discordant pairs is at least 25; otherwise falls back to
    the exact binomial two-sided p-value. Returns ``None`` when no
    matched pairs exist or all pairs are concordant.
    """
    from scipy.stats import binom

    a_map = {(pid, seed): passed for pid, seed, passed in config_a}
    b_map = {(pid, seed): passed for pid, seed, passed in config_b}
    keys = a_map.keys() & b_map.keys()
    if not keys:
        return None

    b = sum(1 for k in keys if a_map[k] and not b_map[k])
    c = sum(1 for k in keys if not a_map[k] and b_map[k])
    n_pairs = len(keys)
    discordant = b + c
    if discordant == 0:
        return McNemarResult(b=b, c=c, n_pairs=n_pairs, statistic=0.0, p_value=1.0, method="exact")

    if discordant < 25:
        k = max(b, c)
        p_value = min(1.0, 2.0 * float(binom.sf(k - 1, discordant, 0.5)))
        return McNemarResult(
            b=b, c=c, n_pairs=n_pairs, statistic=float(k), p_value=p_value, method="exact"
        )

    chi2 = (abs(b - c) - 1) ** 2 / discordant
    from scipy.stats import chi2 as chi2_dist
    p_value = float(chi2_dist.sf(chi2, df=1))
    return McNemarResult(
        b=b, c=c, n_pairs=n_pairs, statistic=float(chi2), p_value=p_value, method="chi2"
    )


def pairwise_mcnemar_table(
    rows: list[dict[str, Any]], out_path: Path
) -> list[dict[str, Any]]:
    """Run McNemar on every ordered pair of configs and write a Markdown table.

    Pairs where no matched (problem_id, seed) runs exist are omitted.
    Results are sorted by p-value ascending. Returns the list of result
    rows (also useful for callers that want to print a summary).
    """
    by_config: dict[str, list[tuple[str, int, bool]]] = defaultdict(list)
    for r in rows:
        if r["error"]:
            continue
        by_config[r["config"]].append((r["problem_id"], r["seed"], r["pass_all_tests"]))

    cfgs = _ordered_configs(set(by_config))
    results: list[dict[str, Any]] = []
    for i, a in enumerate(cfgs):
        for b_cfg in cfgs[i + 1:]:
            res = mcnemar_test(by_config[a], by_config[b_cfg])
            if res is None:
                continue
            row = {"config_a": a, "config_b": b_cfg, **asdict(res)}
            results.append(row)

    results.sort(key=lambda r: r["p_value"])

    header = (
        "| Config A | Config B | n_pairs | b | c | p-value | método |\n"
        "|---|---|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for r in results:
        lines.append(
            f"| {_CONFIG_LABELS.get(r['config_a'], r['config_a'])} | "
            f"{_CONFIG_LABELS.get(r['config_b'], r['config_b'])} | "
            f"{r['n_pairs']} | {r['b']} | {r['c']} | "
            f"{r['p_value']:.4f} | {r['method']} |"
        )
    if len(lines) == 1:
        lines.append("| — | — | — | — | — | — | — |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR_DEFAULT)
    parser.add_argument("--figures-dir", type=Path, default=_FIGURES_DIR_DEFAULT)
    parser.add_argument("--tables-dir", type=Path, default=_TABLES_DIR_DEFAULT)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_results(args.results_dir)
    if not rows:
        print(f"No results found under {args.results_dir}.")
        return

    summary = aggregate_by_config(rows)

    plot_pass_at_1(summary, args.figures_dir / "pass_at_1.png")
    plot_cost_quality(summary, args.figures_dir / "cost_quality_pareto.png")
    plot_latency_box(rows, args.figures_dir / "latency_box.png")
    plot_revision_distribution(rows, args.figures_dir / "revision_distribution.png")

    write_summary_markdown(summary, args.tables_dir / "summary.md")
    write_summary_latex(summary, args.tables_dir / "summary.tex")
    write_per_benchmark_table(rows, args.tables_dir / "per_benchmark.md")
    write_difficulty_report(rows, args.tables_dir / "problem_difficulty.md")
    mcnemar_rows = pairwise_mcnemar_table(
        rows, args.tables_dir / "pairwise_mcnemar.md"
    )
    sig = sum(1 for r in mcnemar_rows if r["p_value"] < 0.05)
    print(
        f"McNemar: {len(mcnemar_rows)} pairs compared, "
        f"{sig} with p<0.05"
    )

    print(f"Wrote figures to {args.figures_dir}/")
    print(f"Wrote tables  to {args.tables_dir}/")
    print(f"Configs analysed: {len(summary)} | total rows: {len(rows)}")


if __name__ == "__main__":
    main()
