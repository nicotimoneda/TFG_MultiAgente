#!/usr/bin/env python3
"""Demo runner for the TFG defense.

Runs the three principal configurations (baseline, sequential,
self_reflection_r1) on a single HumanEval problem, streaming a
human-readable trace to stdout that is suitable for screen recording
(asciinema, OBS, QuickTime), and persists a structured JSON trace under
``demo/trace_<problem_slug>.json`` so the HTML dashboard can render the
same information.

Usage:
    .venv/bin/python experiments/demo_runner.py --problem-id HumanEval/X
        [--model qwen2.5-coder:7b-instruct-q4_K_M]
        [--configs baseline,sequential,self_reflection_r1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.baseline_graph import (
    build_baseline_graph,
    _extract_assert_lines,
)
from src.graph.sequential_graph import build_sequential_graph
from src.graph.self_reflection_graph import build_self_reflection_graph
from src.state.schema import AgentState

_HE_CACHE = Path("experiments/cache/humaneval.json")
_DEMO_DIR = Path("demo")

_BANNER = "=" * 72
_DEFAULT_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
_DEFAULT_CONFIGS = ["baseline", "sequential", "self_reflection_r1"]


def _load_problem(problem_id: str) -> dict:
    data = json.loads(_HE_CACHE.read_text())
    for p in data:
        if p["task_id"] == problem_id:
            return p
    raise SystemExit(f"Problem {problem_id!r} not found in {_HE_CACHE}")


def _initial_state(problem: dict, config_name: str) -> AgentState:
    test_cases = _extract_assert_lines(
        problem.get("test", ""), problem.get("entry_point", "")
    )
    return {
        "problem_id": problem["task_id"],
        "problem_statement": problem["prompt"],
        "function_signature": problem.get("entry_point", ""),
        "test_cases": test_cases,
        "prd": "",
        "design_doc": "",
        "code_artifact": "",
        "test_results": {},
        "review_comments": "",
        "revision_count": 0,
        "tokens_input": 0,
        "tokens_output": 0,
        "latency_seconds": 0.0,
        "config_name": config_name,
    }


def _build_graph(config: str, model: str):
    if config == "baseline":
        return build_baseline_graph(model)
    if config == "sequential":
        return build_sequential_graph(model)
    if config == "self_reflection_r1":
        return build_self_reflection_graph(model_name=model, max_revisions=1)
    raise SystemExit(f"Unknown config {config!r}")


def _preview(text: str, head: int = 3, width: int = 70) -> str:
    if not text:
        return "(empty)"
    lines = [ln.rstrip() for ln in text.strip().splitlines()][:head]
    out = []
    for ln in lines:
        out.append(ln if len(ln) <= width else ln[: width - 1] + "…")
    if len(text.strip().splitlines()) > head:
        out.append("…")
    return "\n      ".join(out)


_ARTIFACT_FIELDS = {
    "pm": "prd",
    "architect": "design_doc",
    "developer": "code_artifact",
    "reviewer": "review_comments",
}


def _stream_config(config: str, problem: dict, model: str) -> dict:
    print()
    print(_BANNER)
    print(f"  CONFIG: {config.upper():<20}  problem: {problem['task_id']}")
    print(_BANNER)

    graph = _build_graph(config, model)
    state = _initial_state(problem, config)

    events = []
    t0 = time.perf_counter()
    prev_in, prev_out = 0, 0
    final_state: dict = dict(state)

    for chunk in graph.stream(state, stream_mode="updates"):
        # chunk: {node_name: state_delta}
        for node, delta in chunk.items():
            if delta is None:
                continue
            final_state.update(delta)
            elapsed = time.perf_counter() - t0
            ti = int(final_state.get("tokens_input", 0))
            to = int(final_state.get("tokens_output", 0))
            d_in, d_out = ti - prev_in, to - prev_out
            prev_in, prev_out = ti, to

            print(
                f"  [{elapsed:7.1f}s] {node:<10s}  +{d_in:>5d} in / +{d_out:>4d} out"
                f"  (acc {ti}/{to})"
            )

            field = _ARTIFACT_FIELDS.get(node)
            if field and field in delta and delta[field]:
                print(f"      {field}:")
                print(f"      {_preview(delta[field])}")
            elif node == "qa" and "test_results" in delta:
                tr = delta["test_results"]
                summary = tr.get("qa_summary") if isinstance(tr, dict) else None
                print(f"      test_results: {summary}")

            events.append(
                {
                    "node": node,
                    "elapsed_s": round(elapsed, 3),
                    "tokens_input_delta": d_in,
                    "tokens_output_delta": d_out,
                    "tokens_input_cum": ti,
                    "tokens_output_cum": to,
                    "artifact_preview": _preview(delta.get(field, ""))
                    if field
                    else None,
                }
            )

    latency = time.perf_counter() - t0
    final_state["latency_seconds"] = latency

    tr = final_state.get("test_results", {}) or {}
    pass_all = bool(tr) and all(
        v for k, v in tr.items() if k != "qa_summary"
    )

    print()
    print(
        f"  RESULT  pass_all_tests={pass_all}  latency={latency:.1f}s"
        f"  tokens={final_state.get('tokens_input', 0)}/"
        f"{final_state.get('tokens_output', 0)}"
        f"  revisions={final_state.get('revision_count', 0)}"
    )

    return {
        "config": config,
        "events": events,
        "pass_all_tests": pass_all,
        "latency_seconds": round(latency, 3),
        "tokens_input": int(final_state.get("tokens_input", 0)),
        "tokens_output": int(final_state.get("tokens_output", 0)),
        "revision_count": int(final_state.get("revision_count", 0)),
        "final_artifacts": {
            "prd": final_state.get("prd", ""),
            "design_doc": final_state.get("design_doc", ""),
            "code_artifact": final_state.get("code_artifact", ""),
            "test_results": final_state.get("test_results", {}),
            "review_comments": final_state.get("review_comments", ""),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", required=True, help="e.g. HumanEval/1")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--configs",
        default=",".join(_DEFAULT_CONFIGS),
        help="Comma-separated subset of baseline,sequential,self_reflection_r1",
    )
    args = parser.parse_args()

    problem = _load_problem(args.problem_id)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    print(_BANNER)
    print(f"  TFG demo — {args.problem_id}")
    print(f"  model: {args.model}")
    print(f"  configs: {', '.join(configs)}")
    print(_BANNER)
    print()
    print("PROBLEM STATEMENT")
    print(_preview(problem["prompt"], head=8, width=70))

    results = []
    for cfg in configs:
        results.append(_stream_config(cfg, problem, args.model))

    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    slug = problem["task_id"].replace("/", "_")
    trace_path = _DEMO_DIR / f"trace_{slug}.json"
    trace_path.write_text(
        json.dumps(
            {
                "problem": {
                    "task_id": problem["task_id"],
                    "entry_point": problem.get("entry_point", ""),
                    "prompt": problem["prompt"],
                    "canonical_solution": problem.get("canonical_solution", ""),
                },
                "model": args.model,
                "configs": {r["config"]: r for r in results},
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(_BANNER)
    print(f"  trace saved: {trace_path}")
    print(_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
