"""Evaluation runner: orchestrates graph execution, sandboxing, and CSV export."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

from src.evaluation.sandbox import execute_code_safely
from src.state.schema import AgentState

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    "benchmark",
    "problem_id",
    "config",
    "seed",
    "pass_all_tests",
    "test_pass_rate",
    "tokens_input",
    "tokens_output",
    "latency_seconds",
    "revision_count",
]


def run_evaluation(
    config: str,
    model_name: str,
    problems: list[dict],
    seeds: list[int],
    output_csv: str,
    max_revisions: int = 1,
) -> None:
    """Run the full evaluation loop and write per-sample results to CSV.

    For each (problem, seed) pair:
      1. Seeds Python's RNG for reproducibility.
      2. Invokes the appropriate graph.
      3. Runs the generated code in the sandbox.
      4. Appends a row to *output_csv*.

    Args:
        config: One of ``'baseline'``, ``'sequential'``, ``'self_reflection'``.
        model_name: Groq model identifier forwarded to the graph.
        problems: List of problem dicts (HumanEval or MBPP schema).
        seeds: List of integer seeds; one full pass per seed.
        output_csv: Path to the output CSV file (created or appended to).
        max_revisions: Maximum self-reflection revision cycles (only used when
            config is ``'self_reflection'``). Defaults to 1.

    Raises:
        ValueError: For unknown config names.
    """
    if config not in ("baseline", "sequential", "self_reflection"):
        raise ValueError(f"Unknown config '{config}'. Expected baseline | sequential | self_reflection.")

    if config == "baseline":
        from src.graph.baseline_graph import run_baseline as _run  # lazy import
    elif config == "sequential":
        from src.graph.sequential_graph import run_sequential as _run  # type: ignore[assignment]  # lazy import
    else:
        from src.graph.self_reflection_graph import run_self_reflection  # lazy import

        def _run(problem: dict, model_name: str) -> AgentState:  # type: ignore[misc]
            return run_self_reflection(problem, model_name, max_revisions=max_revisions)

    n_problems = len(problems)
    total_runs = n_problems * len(seeds)
    run_idx = 0

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()
    csv_file = output_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
    if write_header:
        writer.writeheader()

    try:
        for seed in seeds:
            random.seed(seed)

            for i, problem in enumerate(problems, start=1):
                run_idx += 1
                task_id = problem["task_id"]
                benchmark = "HE" if "HumanEval" in task_id else "MBPP"

                print(
                    f"Running problem {run_idx}/{total_runs} "
                    f"[{benchmark}/{i}] config={config} seed={seed}"
                )

                try:
                    state: AgentState = _run(problem, model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Graph failed for %s seed=%d: %s", task_id, seed, exc)
                    _write_failed_row(writer, benchmark, task_id, config, seed)
                    continue

                # For sequential/self_reflection configs the QA agent already ran the
                # sandbox; re-use those results directly to avoid double execution.
                if config in ("sequential", "self_reflection") and state["test_results"]:
                    raw = {k: v for k, v in state["test_results"].items() if k != "qa_summary"}
                    test_results = raw
                else:
                    test_results = execute_code_safely(
                        code=state["code_artifact"],
                        test_cases=state["test_cases"],
                    )
                pass_all = all(test_results.values()) if test_results else False
                pass_rate = (
                    sum(test_results.values()) / len(test_results)
                    if test_results
                    else 0.0
                )

                writer.writerow(
                    {
                        "benchmark": benchmark,
                        "problem_id": task_id,
                        "config": config,
                        "seed": seed,
                        "pass_all_tests": pass_all,
                        "test_pass_rate": round(pass_rate, 4),
                        "tokens_input": state["tokens_input"],
                        "tokens_output": state["tokens_output"],
                        "latency_seconds": round(state["latency_seconds"], 4),
                        "revision_count": state["revision_count"],
                    }
                )
                csv_file.flush()

    finally:
        csv_file.close()

    logger.info("Evaluation complete. Results written to %s", output_csv)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_failed_row(
    writer: csv.DictWriter,
    benchmark: str,
    task_id: str,
    config: str,
    seed: int,
) -> None:
    """Write a failure placeholder row when graph invocation crashes."""
    writer.writerow(
        {
            "benchmark": benchmark,
            "problem_id": task_id,
            "config": config,
            "seed": seed,
            "pass_all_tests": False,
            "test_pass_rate": 0.0,
            "tokens_input": 0,
            "tokens_output": 0,
            "latency_seconds": 0.0,
            "revision_count": 0,
        }
    )
