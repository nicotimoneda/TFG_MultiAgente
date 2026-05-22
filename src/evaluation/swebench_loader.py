"""SWE-bench Lite loader — scaffolding only.

SWE-bench (Jimenez et al., 2024) evaluates LLM agents on real GitHub issues
from popular open-source Python projects (Django, scikit-learn, Flask, …).
Each task supplies a repository at a specific commit, an issue description
and a hidden test suite that determines whether the proposed patch resolves
the issue.

**This loader fetches the task metadata only.** Actually *running* SWE-bench
requires reproducing the per-repository environment in Docker (the official
harness pulls one image per repository, applies the patch, and runs the test
suite inside the container). That infrastructure is out of scope for this
TFG — it is acknowledged as such in chapter 3 (Alcance y limitaciones) and
chapter 8 (Líneas de trabajo futuro).

The scaffolding is kept in the codebase so a follow-up sprint can integrate
the full SWE-bench Lite run (300 tasks) without reinventing the data layer:
the loader normalises tasks to this project's problem schema (``task_id``,
``prompt``, ``entry_point``, ``test``, ``canonical_solution``) and points
at the official Docker harness for the execution side.

Reference: Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World
GitHub Issues?", ICLR 2024. https://www.swebench.com/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("experiments/cache/swebench_lite.json")
_HF_DATASET = "princeton-nlp/SWE-bench_Lite"
_HF_SPLIT = "test"

# The official harness is required to actually run a SWE-bench task.
# We expose the URL here so the linkage is recorded in the source tree.
SWEBENCH_HARNESS_URL = "https://github.com/princeton-nlp/SWE-bench"


def load_swebench_lite() -> list[dict]:
    """Load the SWE-bench Lite test split with local caching.

    The returned dicts are normalised to this project's schema so the runner
    can consume them by the same code path as HumanEval/MBPP/ClassEval. The
    ``test`` field carries the hidden test patch metadata; **executing** it
    still requires the Docker harness referenced in this module's docstring.
    """
    if _CACHE_PATH.exists():
        logger.info("Loading SWE-bench Lite from cache: %s", _CACHE_PATH)
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    logger.info(
        "Downloading SWE-bench Lite from HuggingFace (%s)…", _HF_DATASET
    )
    from datasets import load_dataset  # lazy import

    dataset = load_dataset(_HF_DATASET, split=_HF_SPLIT)

    problems: list[dict] = []
    for raw in dataset:
        row: dict = dict(raw)  # type: ignore[arg-type]
        problems.append(
            {
                "task_id": f"SWE-bench/{row['instance_id']}",
                "prompt": row.get("problem_statement", ""),
                "entry_point": row.get("repo", ""),
                "test": json.dumps(
                    {
                        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
                        "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
                        "base_commit": row.get("base_commit", ""),
                        "environment_setup_commit": row.get(
                            "environment_setup_commit", ""
                        ),
                    }
                ),
                "canonical_solution": row.get("patch", ""),
            }
        )

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(problems, fh, indent=2)
    logger.info(
        "Cached %d SWE-bench Lite tasks to %s", len(problems), _CACHE_PATH
    )

    return problems


def evaluate_patch(*_args, **_kwargs):  # pragma: no cover - intentional stub
    """Placeholder for the Docker-harness evaluation step.

    Not implemented in this TFG. Wiring this up requires:

    1. Pulling the per-instance Docker image from the official SWE-bench
       harness.
    2. Cloning the target repository at ``base_commit``.
    3. Applying the candidate patch produced by the agent.
    4. Running the ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` test sets inside
       the container and parsing the result.

    See ``SWEBENCH_HARNESS_URL`` for the canonical reference implementation.
    """
    raise NotImplementedError(
        "SWE-bench execution requires the official Docker harness — "
        "this TFG includes only the loader scaffolding. "
        f"See {SWEBENCH_HARNESS_URL} for the runner."
    )
