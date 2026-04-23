"""Evaluation metrics for code-generation benchmarks.

Implements the unbiased pass@k estimator from:
    Chen et al. (2021) "Evaluating Large Language Models Trained on Code"
    https://arxiv.org/abs/2107.03374
"""

from __future__ import annotations

import math
import logging
from typing import Any

from scipy.special import comb  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased pass@k estimator (Chen et al. 2021, eq. 1).

    Formula:
        pass@k = 1 - C(n-c, k) / C(n, k)

    where C(a, b) is the binomial coefficient.  When n < k the estimate is
    undefined; we return 0.0 in that case to avoid division-by-zero.

    Args:
        n: Total number of generated samples for the problem.
        c: Number of those samples that pass all tests (correct samples).
        k: The k in pass@k (e.g. 1, 5, 10).

    Returns:
        Estimated probability that at least one of k randomly chosen samples
        from the n generated is correct.
    """
    if n < k:
        return 0.0
    if n - c < k:
        # All remaining samples after removing correct ones are fewer than k,
        # so picking k is guaranteed to include at least one correct sample.
        return 1.0
    # 1 - C(n-c, k) / C(n, k)  — computed in log space for numerical stability.
    log_numerator = math.lgamma(n - c + 1) - math.lgamma(n - c - k + 1)
    log_denominator = math.lgamma(n + 1) - math.lgamma(n - k + 1)
    return 1.0 - math.exp(log_numerator - log_denominator)


def average_test_pass_rate(results: list[dict[str, bool]]) -> float:
    """Compute the mean fraction of individual test cases passed.

    For each problem, the per-problem pass rate is the fraction of test cases
    that evaluated to True.  This function returns the macro-average across
    all problems.

    Args:
        results: List of ``{test_case_string: passed}`` dicts, one per sample.

    Returns:
        Float in [0, 1].  Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    per_problem_rates = [
        (sum(v for v in r.values()) / len(r)) if r else 0.0
        for r in results
    ]
    return sum(per_problem_rates) / len(per_problem_rates)


def compute_all_metrics(run_results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate all evaluation metrics from a list of run result records.

    Each element of *run_results* should be an AgentState-compatible dict
    (or CSV row dict) with at least:
        - ``test_results``: dict[str, bool]
        - ``tokens_input``: int
        - ``tokens_output``: int
        - ``latency_seconds``: float

    pass@k uses n = total samples per problem, c = number of passing samples.
    For pass@1 with a single seed, this degenerates to 1 if all tests pass.

    Args:
        run_results: List of per-(problem, seed) result dicts.

    Returns:
        Dict with keys: ``pass@1``, ``pass@5``, ``pass@10``,
        ``avg_test_pass_rate``, ``mean_tokens_total``, ``mean_latency_seconds``.
    """
    if not run_results:
        return {
            "pass@1": 0.0,
            "pass@5": 0.0,
            "pass@10": 0.0,
            "avg_test_pass_rate": 0.0,
            "mean_tokens_total": 0.0,
            "mean_latency_seconds": 0.0,
        }

    # Group by problem_id to compute pass@k across seeds.
    from collections import defaultdict

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in run_results:
        by_problem[rec["problem_id"]].append(rec)

    pass_at_1_vals: list[float] = []
    pass_at_5_vals: list[float] = []
    pass_at_10_vals: list[float] = []
    test_result_dicts: list[dict[str, bool]] = []

    for _pid, records in by_problem.items():
        n = len(records)
        c = sum(
            1 for r in records
            if r.get("test_results") and all(r["test_results"].values())
        )
        pass_at_1_vals.append(pass_at_k(n, c, 1))
        pass_at_5_vals.append(pass_at_k(n, c, 5))
        pass_at_10_vals.append(pass_at_k(n, c, 10))

        for r in records:
            if r.get("test_results"):
                test_result_dicts.append(r["test_results"])

    total_tokens = [
        r.get("tokens_input", 0) + r.get("tokens_output", 0)
        for r in run_results
    ]
    latencies = [r.get("latency_seconds", 0.0) for r in run_results]

    return {
        "pass@1": sum(pass_at_1_vals) / len(pass_at_1_vals),
        "pass@5": sum(pass_at_5_vals) / len(pass_at_5_vals),
        "pass@10": sum(pass_at_10_vals) / len(pass_at_10_vals),
        "avg_test_pass_rate": average_test_pass_rate(test_result_dicts),
        "mean_tokens_total": sum(total_tokens) / len(total_tokens),
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }
