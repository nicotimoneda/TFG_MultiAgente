"""ClassEval dataset loader with local caching.

ClassEval (Du et al., 2023) extends HumanEval-style function generation to
**class-level** code generation: each problem provides a class skeleton with
multiple methods that must be implemented coherently, plus a test suite that
exercises the public interface. It is the smallest of the multi-file-flavoured
benchmarks mentioned in the propuesta and the one that fits this project's
infrastructure with the fewest modifications.

The loader follows the same pattern as ``humaneval_loader``: first call hits
HuggingFace and writes a cache to ``experiments/cache/classeval.json``; later
calls read directly from disk so the experiment can run offline.

Note on integration with the existing pipeline. The current AgentState carries
a single ``code_artifact`` string and a flat ``test_cases`` list. ClassEval
problems generate **classes** rather than functions; the harness packs the
class source into ``code_artifact`` and concatenates the canonical test
harness into the sandbox invocation. Running the full ClassEval suite on the
multi-agent pipeline therefore requires:

1. Prompts updated to instruct the Developer to emit a class definition
   inside the ``python`` fence (instead of a function).
2. Sandbox invocation that imports the class and runs the per-class test
   driver provided by ClassEval.

Both extensions are recorded as future work in chapter 8; this loader makes
the data available so they can be implemented in a follow-up sprint without
having to redo the data plumbing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("experiments/cache/classeval.json")
_HF_DATASET = "FudanSELab/ClassEval"
_HF_SPLIT = "test"


def load_classeval() -> list[dict]:
    """Load every ClassEval problem, using a local cache when available.

    Returns:
        List of problem dicts in this project's normalized schema:
        ``task_id``, ``prompt``, ``entry_point``, ``test``, ``canonical_solution``.

        For ClassEval, ``entry_point`` is the class name. ``test`` is the
        full pytest-style test module supplied by the benchmark and must be
        executed inside the sandbox after defining the class.
    """
    if _CACHE_PATH.exists():
        logger.info("Loading ClassEval from cache: %s", _CACHE_PATH)
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Downloading ClassEval from HuggingFace (%s)…", _HF_DATASET)
    from datasets import load_dataset  # lazy import — optional at module load

    dataset = load_dataset(_HF_DATASET, split=_HF_SPLIT)

    problems: list[dict] = []
    for raw in dataset:
        row: dict = dict(raw)  # type: ignore[arg-type]
        problems.append(
            {
                "task_id": f"ClassEval/{row['task_id']}",
                "prompt": row.get("skeleton") or row.get("class_description", ""),
                "entry_point": row.get("class_name", ""),
                "test": row.get("test", ""),
                "canonical_solution": row.get("solution_code", ""),
            }
        )

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(problems, fh, indent=2)
    logger.info("Cached %d ClassEval problems to %s", len(problems), _CACHE_PATH)

    return problems


def get_problem(task_id: str) -> dict:
    """Return a single ClassEval problem by task_id (KeyError if missing)."""
    problems = load_classeval()
    index = {p["task_id"]: p for p in problems}
    if task_id not in index:
        raise KeyError(f"task_id {task_id!r} not found in ClassEval.")
    return index[task_id]
