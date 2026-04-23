"""HumanEval dataset loader with local caching.

Downloads the openai/human-eval dataset from HuggingFace on first use and
caches it to disk so subsequent runs are offline-capable.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("experiments/cache/humaneval.json")
_HF_DATASET = "openai/human-eval"
_HF_SPLIT = "test"


def load_humaneval() -> list[dict]:
    """Load all 164 HumanEval problems, using a local cache when available.

    On the first call the dataset is fetched from HuggingFace and written to
    ``experiments/cache/humaneval.json``. Subsequent calls read from that file.

    Returns:
        List of problem dicts, each containing:
        ``task_id``, ``prompt``, ``entry_point``, ``test``, ``canonical_solution``.
    """
    if _CACHE_PATH.exists():
        logger.info("Loading HumanEval from cache: %s", _CACHE_PATH)
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Downloading HumanEval from HuggingFace (%s)…", _HF_DATASET)
    from datasets import load_dataset  # lazy import — optional at module load time

    dataset = load_dataset(_HF_DATASET, split=_HF_SPLIT, trust_remote_code=True)

    problems: list[dict] = [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "canonical_solution": row["canonical_solution"],
        }
        for row in dataset
    ]

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(problems, fh, indent=2)
    logger.info("Cached %d problems to %s", len(problems), _CACHE_PATH)

    return problems


def get_problem(task_id: str) -> dict:
    """Return a single HumanEval problem by its task_id.

    Args:
        task_id: Identifier such as ``'HumanEval/0'``.

    Returns:
        Problem dict with keys ``task_id``, ``prompt``, ``entry_point``,
        ``test``, ``canonical_solution``.

    Raises:
        KeyError: If ``task_id`` is not found in the dataset.
    """
    problems = load_humaneval()
    index = {p["task_id"]: p for p in problems}
    if task_id not in index:
        raise KeyError(f"task_id '{task_id}' not found in HumanEval.")
    return index[task_id]
