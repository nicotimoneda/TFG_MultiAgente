#!/usr/bin/env python3
r"""Structured-output adherence metric for the multi-agent pipeline.

The propuesta claims that a structured artifact protocol (PRD, design doc,
code fences) reduces hallucinations vs. free-form conversation. This script
operationalises that claim by counting, per configuration, how often each
agent failed to emit a well-formed structured artifact.

Signals used (all already logged by the agents):

* ``No \`\`\`python\`\`\` fence found in response``     — baseline / developer
* ``No \`\`\`python\`\`\` fence in developer response`` — sequential developer
* ``No \`\`\`python\`\`\` fence in reflective developer response`` — SR developer

For each run (= one CSV row) we know which warnings happened by matching the
problem_id mentioned in the log line. Adherence is then::

    adherence = 1 - (#runs_with_any_structural_warning / total_runs)

The metric is post-hoc and read-only: it never re-invokes the LLM and does
not modify experimental state.

Outputs a markdown table + a per-config JSON summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import csv

_LOG_DEFAULT = Path("experiments/logs/run.out")
_RESULTS_DEFAULT = Path("experiments/results")
_OUT_TABLE = Path("doc/tables/adherence.md")
_OUT_JSON = Path("experiments/results/adherence.json")

# A warning line carries the problem_id at the end. We don't know which config
# was active from the warning alone, so we correlate against the per-CSV
# (problem_id, seed) tuples — every CSV row maps 1-to-N to warnings emitted
# during its window. We approximate: count any warning line that names a
# problem_id and attribute it to whichever configs ran that problem_id.
#
# Since the runner now logs the [n/total] OK line *after* the run, the
# warnings emitted between two OK lines belong to that completed run. We
# walk the log linearly and track the "current run" by parsing those OK lines.

_RUN_BOUNDARY_RE = re.compile(
    r"^\[INFO\] \[\d+/\d+\]\s+(?P<pid>\S+)\s+config=(?P<cfg>\S+)\s+seed=(?P<seed>\d+)\s+→"
)
_WARN_FENCE_RE = re.compile(
    r"No ```python``` fence (?:found in response|in developer response|"
    r"in reflective developer response)"
)
_CURRENT_RE = re.compile(
    r'"current":\s*\{\s*"benchmark":\s*"(?P<bm>[^"]+)",\s*"problem_id":'
    r'\s*"(?P<pid>[^"]+)",\s*"config":\s*"(?P<cfg>[^"]+)",\s*"seed":\s*(?P<seed>\d+)'
)


def _iter_run_windows(log_path: Path) -> Iterable[tuple[str, str, int, list[str]]]:
    """Yield (problem_id, config, seed, warnings) for each completed run."""
    if not log_path.exists():
        return
    pending: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _RUN_BOUNDARY_RE.match(line)
            if m:
                # The pending warnings belong to the run that just closed.
                yield (
                    m.group("pid"),
                    m.group("cfg"),
                    int(m.group("seed")),
                    pending,
                )
                pending = []
                continue
            if "WARNING" in line and _WARN_FENCE_RE.search(line):
                pending.append(line)


def _count_total_runs(results_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        if csv_path.name == "quick_check.csv":
            continue
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("error", "").strip():
                    continue
                totals[row["config"]] += 1
    return dict(totals)


def compute_adherence(
    log_path: Path, results_dir: Path
) -> dict[str, dict[str, float | int]]:
    """Compute per-config adherence rates from log + CSVs."""
    totals = _count_total_runs(results_dir)
    fence_failures: dict[str, int] = defaultdict(int)
    runs_with_any_warning: dict[str, int] = defaultdict(int)

    for _pid, cfg, _seed, warnings in _iter_run_windows(log_path):
        if not warnings:
            continue
        fence_failures[cfg] += len(warnings)
        runs_with_any_warning[cfg] += 1

    summary: dict[str, dict[str, float | int]] = {}
    for cfg, n in totals.items():
        warned = runs_with_any_warning.get(cfg, 0)
        summary[cfg] = {
            "total_runs": n,
            "runs_with_structural_warning": warned,
            "total_structural_warnings": fence_failures.get(cfg, 0),
            "adherence_rate": 1.0 - (warned / n) if n else 0.0,
        }
    return summary


def write_markdown(summary: dict[str, dict[str, float | int]], out: Path) -> None:
    header = (
        "| Configuración | Runs | Runs con fallo estructural | "
        "Avisos totales | Adherencia |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for cfg in sorted(summary):
        s = summary[cfg]
        lines.append(
            f"| {cfg} | {s['total_runs']} | "
            f"{s['runs_with_structural_warning']} | "
            f"{s['total_structural_warnings']} | "
            f"{float(s['adherence_rate']):.2%} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=_LOG_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DEFAULT)
    parser.add_argument("--out-table", type=Path, default=_OUT_TABLE)
    parser.add_argument("--out-json", type=Path, default=_OUT_JSON)
    args = parser.parse_args()

    summary = compute_adherence(args.log, args.results_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.out_table)
    print(f"Adherence summary written to {args.out_table} and {args.out_json}")
    for cfg, s in sorted(summary.items()):
        print(
            f"  {cfg:30s} runs={s['total_runs']:5d} "
            f"warned={s['runs_with_structural_warning']:5d} "
            f"adherence={float(s['adherence_rate']):.2%}"
        )


if __name__ == "__main__":
    main()
