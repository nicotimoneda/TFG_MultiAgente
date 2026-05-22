"""Tests for the analysis pipeline.

The analysis pipeline produces the numbers that go into chapter 7 and the
executive summary, so silent bugs here have outsized impact.
"""

from __future__ import annotations


import pytest

from experiments.analyze_results import (
    _coerce_row,
    _pass_at_k,
    aggregate_by_config,
    bootstrap_pass_at_1_ci,
    mcnemar_test,
)


# ---------------------------------------------------------------------------
# _pass_at_k
# ---------------------------------------------------------------------------

def test_pass_at_k_unbiased_estimator_formula():
    # Reference: Chen et al. (2021) eq. 1, identity for k = 1.
    assert _pass_at_k(n=3, c=2, k=1) == pytest.approx(2 / 3)
    assert _pass_at_k(n=3, c=0, k=1) == pytest.approx(0.0)
    assert _pass_at_k(n=3, c=3, k=1) == pytest.approx(1.0)


def test_pass_at_k_saturates_when_few_failures():
    assert _pass_at_k(n=5, c=4, k=3) == pytest.approx(1.0)


def test_pass_at_k_zero_when_n_less_than_k():
    assert _pass_at_k(n=2, c=2, k=5) == 0.0


# ---------------------------------------------------------------------------
# bootstrap_pass_at_1_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_deterministic_with_fixed_seed():
    passes = [True] * 80 + [False] * 20
    low_a, high_a = bootstrap_pass_at_1_ci(passes, n_boot=500, seed=42)
    low_b, high_b = bootstrap_pass_at_1_ci(passes, n_boot=500, seed=42)
    assert low_a == low_b
    assert high_a == high_b


def test_bootstrap_ci_brackets_point_estimate():
    passes = [True] * 80 + [False] * 20
    low, high = bootstrap_pass_at_1_ci(passes, n_boot=2000, seed=0)
    # Point estimate is 0.80; CI should contain it (with very high probability)
    assert low <= 0.80 <= high
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_ci_empty_input():
    assert bootstrap_pass_at_1_ci([], n_boot=100, seed=0) == (0.0, 0.0)


def test_bootstrap_ci_all_success():
    low, high = bootstrap_pass_at_1_ci([True] * 50, n_boot=500, seed=0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _coerce_row
# ---------------------------------------------------------------------------

def _row_template(**overrides):
    base = {
        "benchmark": "HE",
        "problem_id": "HumanEval/0",
        "config": "baseline",
        "seed": "42",
        "pass_all_tests": "True",
        "test_pass_rate": "1.0",
        "tokens_input": "100",
        "tokens_output": "50",
        "latency_seconds": "2.5",
        "revision_count": "0",
        "error": "",
    }
    base.update(overrides)
    return base


def test_coerce_row_happy_path():
    out = _coerce_row(_row_template())
    assert out is not None
    assert out["pass_all_tests"] is True
    assert out["seed"] == 42
    assert out["tokens_input"] == 100
    assert out["latency_seconds"] == 2.5


def test_coerce_row_false_pass():
    out = _coerce_row(_row_template(pass_all_tests="False"))
    assert out is not None
    assert out["pass_all_tests"] is False


def test_coerce_row_empty_numeric_fields():
    out = _coerce_row(_row_template(test_pass_rate="", tokens_input=""))
    assert out is not None
    assert out["test_pass_rate"] == 0.0
    assert out["tokens_input"] == 0


def test_coerce_row_malformed_returns_none():
    bad = _row_template()
    bad["seed"] = "not-a-number"
    assert _coerce_row(bad) is None


# ---------------------------------------------------------------------------
# aggregate_by_config
# ---------------------------------------------------------------------------

def _make_rows():
    return [
        {
            "benchmark": "HE",
            "problem_id": f"HumanEval/{i % 3}",
            "config": "baseline" if i < 6 else "sequential",
            "seed": (i % 3) * 100 + 42,
            "pass_all_tests": (i % 2 == 0),
            "test_pass_rate": 1.0 if (i % 2 == 0) else 0.5,
            "tokens_input": 100,
            "tokens_output": 50 * (i + 1),
            "latency_seconds": 1.0 + i * 0.1,
            "revision_count": 0,
            "error": "",
        }
        for i in range(12)
    ]


def test_aggregate_by_config_counts_per_group():
    summary = aggregate_by_config(_make_rows())
    assert "baseline" in summary
    assert "sequential" in summary
    assert summary["baseline"]["n_runs"] == 6
    assert summary["sequential"]["n_runs"] == 6


def test_aggregate_by_config_skips_error_rows():
    rows = _make_rows()
    rows[0]["error"] = "boom"
    summary = aggregate_by_config(rows)
    assert summary["baseline"]["n_runs"] == 5


def test_aggregate_by_config_pass_at_1_matches_proportion():
    summary = aggregate_by_config(_make_rows())
    # By construction, half pass.
    assert summary["baseline"]["pass_at_1"] == pytest.approx(0.5)


def test_aggregate_by_config_includes_bootstrap_ci_fields():
    summary = aggregate_by_config(_make_rows())
    s = summary["baseline"]
    assert "pass_at_1_ci_low" in s
    assert "pass_at_1_ci_high" in s
    assert s["pass_at_1_ci_low"] <= s["pass_at_1"] <= s["pass_at_1_ci_high"]


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------

def _matched_rows(passes_a, passes_b, problems=None):
    """Build two (problem_id, seed, passed) tuple-lists sharing keys."""
    n = len(passes_a)
    assert len(passes_b) == n
    if problems is None:
        problems = [(f"P/{i}", i) for i in range(n)]
    rows_a = [(pid, s, pa) for (pid, s), pa in zip(problems, passes_a)]
    rows_b = [(pid, s, pb) for (pid, s), pb in zip(problems, passes_b)]
    return rows_a, rows_b


def test_mcnemar_exact_when_few_discordant():
    # b = 3, c = 0: all 3 discordant pairs favour A; exact branch.
    a, b = _matched_rows(
        [True, True, True, True, False],
        [False, False, False, True, False],
    )
    res = mcnemar_test(a, b)
    assert res is not None
    assert res.b == 3
    assert res.c == 0
    assert res.method == "exact"
    # Exact two-sided p-value for (b+c=3, max=3): 2 * P(X>=3) = 2 * 0.125 = 0.25
    assert res.p_value == pytest.approx(0.25, abs=1e-9)


def test_mcnemar_chi_squared_when_many_discordant():
    # b=20, c=10 → b+c=30 (>=25) → chi² branch
    passes_a = [True] * 20 + [False] * 10 + [True] * 5 + [False] * 5
    passes_b = [False] * 20 + [True] * 10 + [True] * 5 + [False] * 5
    a, b = _matched_rows(passes_a, passes_b)
    res = mcnemar_test(a, b)
    assert res is not None
    assert res.b == 20
    assert res.c == 10
    assert res.method == "chi2"
    # chi² = (|20-10|-1)² / (20+10) = 81/30 = 2.7
    assert res.statistic == pytest.approx(2.7, abs=1e-9)


def test_mcnemar_returns_none_when_no_overlap():
    a = [("P/1", 1, True)]
    b = [("P/2", 2, False)]
    assert mcnemar_test(a, b) is None
