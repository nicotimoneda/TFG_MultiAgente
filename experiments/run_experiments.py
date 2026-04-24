#!/usr/bin/env python3
"""Full experiment runner for TFG MultiAgente.

Usage:
    python experiments/run_experiments.py [--model MODEL] [--configs a,b,c]
        [--benchmarks humaneval,mbpp] [--seeds 42,123] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Load .env (inline, no python-dotenv required)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path("experiments/results")
_LOGS_DIR = Path("experiments/logs")
_PROGRESS_FILE = _RESULTS_DIR / "progress.json"
_MBPP_CACHE = Path("experiments/cache/mbpp.json")

_ALL_SEEDS = [42, 123, 456, 789, 1234]
_DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

# Each entry: (config_key, runner_config, max_revisions, csv_path)
_CONFIG_SPECS: list[tuple[str, str, int, Path]] = [
    ("baseline",          "baseline",        0, _RESULTS_DIR / "baseline_results.csv"),
    ("sequential",        "sequential",       0, _RESULTS_DIR / "sequential_results.csv"),
    ("self_reflection_r1","self_reflection",  1, _RESULTS_DIR / "self_reflection_r1_results.csv"),
    ("self_reflection_r2","self_reflection",  2, _RESULTS_DIR / "self_reflection_r2_results.csv"),
    ("self_reflection_r3","self_reflection",  3, _RESULTS_DIR / "self_reflection_r3_results.csv"),
]

_CSV_FIELDS = [
    "benchmark", "problem_id", "config", "seed",
    "pass_all_tests", "test_pass_rate",
    "tokens_input", "tokens_output", "latency_seconds",
    "revision_count", "timestamp", "model", "error",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(_LOGS_DIR / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[error_handler, console_handler])

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MBPP loader
# ---------------------------------------------------------------------------

def _extract_entry_point(code: str) -> str:
    match = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return match.group(1) if match else "solution"


def load_mbpp(n: int = 200) -> list[dict]:
    if _MBPP_CACHE.exists():
        logger.info("Loading MBPP from cache: %s", _MBPP_CACHE)
        with _MBPP_CACHE.open("r", encoding="utf-8") as fh:
            return json.load(fh)[:n]

    logger.info("Downloading MBPP from HuggingFace…")
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset("google-research-datasets/mbpp", split="train")

    problems: list[dict] = []
    for row in dataset:
        code = row.get("code", "")
        entry_point = _extract_entry_point(code)
        test_lines = row.get("test_list", []) or []
        setup = row.get("test_setup_code", "") or ""
        test_body = (setup + "\n" if setup else "") + "\n".join(test_lines)
        problems.append(
            {
                "task_id": f"MBPP/{row['task_id']}",
                "prompt": row.get("text", ""),
                "entry_point": entry_point,
                "test": test_body,
                "canonical_solution": code,
            }
        )

    _MBPP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _MBPP_CACHE.open("w", encoding="utf-8") as fh:
        json.dump(problems, fh, indent=2)
    logger.info("Cached %d MBPP problems to %s", len(problems), _MBPP_CACHE)

    return problems[:n]

# ---------------------------------------------------------------------------
# Resume: load completed (benchmark, problem_id, config, seed) tuples
# ---------------------------------------------------------------------------

def _load_completed(csv_path: Path, config_key: str) -> set[tuple[str, str, str, int]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, str, str, int]] = set()
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("error", ""):
                    # Skip failed rows so they can be retried
                    continue
                try:
                    done.add(
                        (
                            row["benchmark"],
                            row["problem_id"],
                            row["config"],
                            int(row["seed"]),
                        )
                    )
                except (KeyError, ValueError):
                    pass
    except Exception:
        logger.warning("Could not read resume data from %s", csv_path)
    return done

# ---------------------------------------------------------------------------
# Progress.json (atomic write)
# ---------------------------------------------------------------------------

_progress_lock = threading.Lock()
_progress: dict = {
    "total": 0,
    "completed": 0,
    "failed": 0,
    "last_updated": "",
    "current": {},
}


def _save_progress() -> None:
    _progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = _PROGRESS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_progress, indent=2), encoding="utf-8")
    tmp.replace(_PROGRESS_FILE)


def _update_progress(
    *,
    completed_delta: int = 0,
    failed_delta: int = 0,
    current: dict | None = None,
) -> None:
    with _progress_lock:
        _progress["completed"] += completed_delta
        _progress["failed"] += failed_delta
        if current is not None:
            _progress["current"] = current
        _save_progress()

# ---------------------------------------------------------------------------
# CSV writers (one per config, thread-safe)
# ---------------------------------------------------------------------------

class _CsvWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_FIELDS)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def writerow(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

# ---------------------------------------------------------------------------
# Graph runners (lazy-imported)
# ---------------------------------------------------------------------------

def _run_problem(
    *,
    runner_config: str,
    max_revisions: int,
    problem: dict,
    model_name: str,
) -> dict:
    from src.state.schema import AgentState  # type: ignore

    if runner_config == "baseline":
        from src.graph.baseline_graph import run_baseline  # type: ignore
        return run_baseline(problem, model_name)  # type: ignore[return-value]
    elif runner_config == "sequential":
        from src.graph.sequential_graph import run_sequential  # type: ignore
        return run_sequential(problem, model_name)  # type: ignore[return-value]
    else:
        from src.graph.self_reflection_graph import run_self_reflection  # type: ignore
        return run_self_reflection(problem, model_name, max_revisions=max_revisions)  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Per-run task
# ---------------------------------------------------------------------------

def _execute_run(
    *,
    config_key: str,
    runner_config: str,
    max_revisions: int,
    problem: dict,
    seed: int,
    model_name: str,
    writer: _CsvWriter,
    benchmark: str,
) -> bool:
    task_id = problem["task_id"]
    random.seed(seed)

    _update_progress(
        current={
            "benchmark": benchmark,
            "problem_id": task_id,
            "config": config_key,
            "seed": seed,
        }
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        state = _run_problem(
            runner_config=runner_config,
            max_revisions=max_revisions,
            problem=problem,
            model_name=model_name,
        )

        # For sequential/self_reflection, QA already ran the sandbox.
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
            if test_results
            else 0.0
        )

        writer.writerow(
            {
                "benchmark": benchmark,
                "problem_id": task_id,
                "config": config_key,
                "seed": seed,
                "pass_all_tests": pass_all,
                "test_pass_rate": round(pass_rate, 4),
                "tokens_input": state["tokens_input"],  # type: ignore[index]
                "tokens_output": state["tokens_output"],  # type: ignore[index]
                "latency_seconds": round(state["latency_seconds"], 4),  # type: ignore[index]
                "revision_count": state["revision_count"],  # type: ignore[index]
                "timestamp": timestamp,
                "model": model_name,
                "error": "",
            }
        )
        _update_progress(completed_delta=1)
        return True

    except Exception as exc:
        err_str = str(exc)[:200]
        logger.error(
            "FAILED %s config=%s seed=%d: %s\n%s",
            task_id, config_key, seed, err_str,
            traceback.format_exc(),
        )
        writer.writerow(
            {
                "benchmark": benchmark,
                "problem_id": task_id,
                "config": config_key,
                "seed": seed,
                "pass_all_tests": False,
                "test_pass_rate": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "latency_seconds": 0.0,
                "revision_count": 0,
                "timestamp": timestamp,
                "model": model_name,
                "error": err_str,
            }
        )
        _update_progress(failed_delta=1)
        return False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFG MultiAgente experiment runner")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="HuggingFace model repo ID")
    parser.add_argument(
        "--configs",
        default="baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3",
        help="Comma-separated config keys",
    )
    parser.add_argument(
        "--benchmarks",
        default="humaneval,mbpp",
        help="Comma-separated benchmarks: humaneval, mbpp",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in _ALL_SEEDS),
        help="Comma-separated integer seeds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default 1 — safe for API rate limits)",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()

    requested_configs = [c.strip() for c in args.configs.split(",")]
    requested_benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    model_name = args.model
    workers = max(1, args.workers)

    # Filter config specs to requested configs
    config_specs = [s for s in _CONFIG_SPECS if s[0] in requested_configs]
    if not config_specs:
        logger.error("No valid configs in: %s", args.configs)
        sys.exit(1)

    # Load benchmarks
    problems_by_benchmark: dict[str, list[dict]] = {}
    if "humaneval" in requested_benchmarks:
        from src.evaluation.humaneval_loader import load_humaneval  # type: ignore
        problems_by_benchmark["humaneval"] = load_humaneval()
        logger.info("Loaded %d HumanEval problems", len(problems_by_benchmark["humaneval"]))
    if "mbpp" in requested_benchmarks:
        problems_by_benchmark["mbpp"] = load_mbpp(200)
        logger.info("Loaded %d MBPP problems", len(problems_by_benchmark["mbpp"]))

    # Build full task list, excluding already-completed runs
    writers: dict[str, _CsvWriter] = {}
    tasks: list[dict] = []

    for config_key, runner_config, max_revisions, csv_path in config_specs:
        completed = _load_completed(csv_path, config_key)
        writer = _CsvWriter(csv_path)
        writers[config_key] = writer

        for bmark, problems in problems_by_benchmark.items():
            bmark_label = "HE" if bmark == "humaneval" else "MBPP"
            for problem in problems:
                task_id = problem["task_id"]
                for seed in seeds:
                    key = (bmark_label, task_id, config_key, seed)
                    if key in completed:
                        continue
                    tasks.append(
                        {
                            "config_key": config_key,
                            "runner_config": runner_config,
                            "max_revisions": max_revisions,
                            "problem": problem,
                            "seed": seed,
                            "model_name": model_name,
                            "writer": writer,
                            "benchmark": bmark_label,
                        }
                    )

    total_new = len(tasks)
    already_done = sum(
        len(_load_completed(csv_path, ck)) for ck, _, _, csv_path in config_specs
    )
    total_all = already_done + total_new

    with _progress_lock:
        _progress["total"] = total_all
        _progress["completed"] = already_done
        _progress["failed"] = 0
        _save_progress()

    logger.info(
        "Experiment matrix: %d total | %d already done | %d to run",
        total_all, already_done, total_new,
    )

    if not tasks:
        logger.info("All runs already complete. Nothing to do.")
        for w in writers.values():
            w.close()
        return

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_run, **task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    ok = future.result()
                    status = "OK" if ok else "FAIL"
                except Exception as exc:
                    status = f"ERROR: {exc}"
                logger.info(
                    "[%d/%d] %s config=%s seed=%d → %s",
                    already_done + i, total_all,
                    task["problem"]["task_id"],
                    task["config_key"],
                    task["seed"],
                    status,
                )
    finally:
        for w in writers.values():
            w.close()

    logger.info("Done. Results in %s", _RESULTS_DIR)


if __name__ == "__main__":
    main()
