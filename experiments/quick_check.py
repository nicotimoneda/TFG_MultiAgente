#!/usr/bin/env python3
"""Quick sanity check: 10 HumanEval problems × 5 configs × seed=42.

Expected runtime: ~10-15 minutes.
Writes results to experiments/results/quick_check.csv and prints a summary table.

Usage:
    python experiments/quick_check.py
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

_PROBLEM_IDS = [
    "HumanEval/1",
    "HumanEval/5",
    "HumanEval/10",
    "HumanEval/26",
    "HumanEval/38",
    "HumanEval/50",
    "HumanEval/75",
    "HumanEval/100",
    "HumanEval/119",
    "HumanEval/150",
]

_CONFIGS: list[tuple[str, str, int]] = [
    ("baseline",           "baseline",       0),
    ("sequential",         "sequential",     0),
    ("self_reflection_r1", "self_reflection", 1),
    ("self_reflection_r2", "self_reflection", 2),
    ("self_reflection_r3", "self_reflection", 3),
]

_SEED = 42
_OUT_CSV = Path("experiments/results/quick_check.csv")
_MODEL = "qwen-3-235b-a22b-instruct-2507"

_CSV_FIELDS = [
    "benchmark", "problem_id", "config", "seed",
    "pass_all_tests", "test_pass_rate",
    "tokens_input", "tokens_output", "latency_seconds",
    "revision_count", "timestamp", "model", "error",
]


def _run_one(runner_config: str, max_revisions: int, problem: dict) -> dict:
    if runner_config == "baseline":
        from src.graph.baseline_graph import run_baseline  # type: ignore
        return run_baseline(problem, _MODEL)  # type: ignore[return-value]
    elif runner_config == "sequential":
        from src.graph.sequential_graph import run_sequential  # type: ignore
        return run_sequential(problem, _MODEL)  # type: ignore[return-value]
    else:
        from src.graph.self_reflection_graph import run_self_reflection  # type: ignore
        return run_self_reflection(problem, _MODEL, max_revisions=max_revisions)  # type: ignore[return-value]


def main() -> None:
    from src.evaluation.humaneval_loader import get_problem  # type: ignore

    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    problems = [get_problem(pid) for pid in _PROBLEM_IDS]
    total = len(problems) * len(_CONFIGS)
    results: list[dict] = []

    print(f"\nQuick check: {len(problems)} problems × {len(_CONFIGS)} configs × seed={_SEED}")
    print(f"Total runs: {total}\n")

    run_idx = 0
    with _OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for config_key, runner_config, max_revisions in _CONFIGS:
            for problem in problems:
                run_idx += 1
                task_id = problem["task_id"]
                print(f"  [{run_idx:2d}/{total}] {task_id:<20} config={config_key}", end=" ", flush=True)

                random.seed(_SEED)
                timestamp = datetime.now(timezone.utc).isoformat()

                try:
                    state = _run_one(runner_config, max_revisions, problem)

                    if runner_config in ("sequential", "self_reflection") and state.get("test_results"):  # type: ignore[union-attr]
                        raw = {k: v for k, v in state["test_results"].items() if k != "qa_summary"}  # type: ignore[index]
                        test_results = raw
                    else:
                        from src.evaluation.sandbox import execute_code_safely  # type: ignore
                        test_results = execute_code_safely(
                            code=state["code_artifact"],  # type: ignore[index]
                            test_cases=state["test_cases"],  # type: ignore[index]
                        )

                    pass_all = all(test_results.values()) if test_results else False
                    pass_rate = (
                        sum(1 for v in test_results.values() if v) / len(test_results)
                        if test_results else 0.0
                    )
                    status = "✓ PASS" if pass_all else f"✗ {pass_rate:.0%}"
                    print(f"→ {status} | tokens={state['tokens_input'] + state['tokens_output']} | {state['latency_seconds']:.1f}s")  # type: ignore[index]

                    row = {
                        "benchmark": "HE",
                        "problem_id": task_id,
                        "config": config_key,
                        "seed": _SEED,
                        "pass_all_tests": pass_all,
                        "test_pass_rate": round(pass_rate, 4),
                        "tokens_input": state["tokens_input"],  # type: ignore[index]
                        "tokens_output": state["tokens_output"],  # type: ignore[index]
                        "latency_seconds": round(state["latency_seconds"], 4),  # type: ignore[index]
                        "revision_count": state["revision_count"],  # type: ignore[index]
                        "timestamp": timestamp,
                        "model": _MODEL,
                        "error": "",
                    }

                except Exception as exc:
                    print(f"→ ERROR: {exc}")
                    row = {
                        "benchmark": "HE",
                        "problem_id": task_id,
                        "config": config_key,
                        "seed": _SEED,
                        "pass_all_tests": False,
                        "test_pass_rate": 0.0,
                        "tokens_input": 0,
                        "tokens_output": 0,
                        "latency_seconds": 0.0,
                        "revision_count": 0,
                        "timestamp": timestamp,
                        "model": _MODEL,
                        "error": str(exc)[:200],
                    }
                    pass_all = False
                    pass_rate = 0.0

                writer.writerow(row)
                fh.flush()
                results.append(row)

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    header = f"{'Config':<22} {'Passes':>7} {'Total':>6} {'Pass%':>7} {'Avg Tokens':>11} {'Avg Lat':>9}"
    print(header)
    print("-" * 80)

    for config_key, _, _ in _CONFIGS:
        config_rows = [r for r in results if r["config"] == config_key]
        passes = sum(1 for r in config_rows if str(r["pass_all_tests"]).lower() in ("true", "1"))
        total_c = len(config_rows)
        avg_tok = (
            sum(int(r["tokens_input"]) + int(r["tokens_output"]) for r in config_rows) / total_c
            if total_c else 0
        )
        avg_lat = sum(float(r["latency_seconds"]) for r in config_rows) / total_c if total_c else 0
        pct = passes / total_c if total_c else 0
        print(
            f"{config_key:<22} {passes:>7} {total_c:>6} {pct:>7.1%} {avg_tok:>11,.0f} {avg_lat:>8.1f}s"
        )

    print("=" * 80)
    print(f"\nFull results written to: {_OUT_CSV}")


if __name__ == "__main__":
    main()
