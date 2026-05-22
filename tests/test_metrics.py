"""Tests for the evaluation metrics module.

pass@k is the central scoreboard of the experiment. A buggy estimator would
silently bias every number reported in chapter 7.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import pass_at_k, average_test_pass_rate


def test_pass_at_1_all_correct():
    assert pass_at_k(n=3, c=3, k=1) == pytest.approx(1.0)


def test_pass_at_1_none_correct():
    assert pass_at_k(n=3, c=0, k=1) == pytest.approx(0.0)


def test_pass_at_1_half_correct():
    # n=4, c=2 → pass@1 = 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
    assert pass_at_k(n=4, c=2, k=1) == pytest.approx(0.5)


def test_pass_at_k_underflow_when_n_lt_k():
    assert pass_at_k(n=2, c=2, k=5) == 0.0


def test_pass_at_k_saturates_when_few_failures():
    # If n - c < k, every k-subset is guaranteed to include a correct sample.
    assert pass_at_k(n=5, c=4, k=3) == pytest.approx(1.0)


def test_pass_at_k_matches_chen_formula_for_small_n():
    # Chen et al. (2021) eq. 1: pass@k = 1 - C(n-c, k) / C(n, k)
    # n=10, c=3, k=3 → 1 - C(7,3)/C(10,3) = 1 - 35/120 = 17/24
    assert pass_at_k(n=10, c=3, k=3) == pytest.approx(17 / 24)


def test_average_test_pass_rate_empty():
    assert average_test_pass_rate([]) == 0.0


def test_average_test_pass_rate_all_pass():
    results = [{"t1": True, "t2": True}, {"t1": True}]
    assert average_test_pass_rate(results) == pytest.approx(1.0)


def test_average_test_pass_rate_mixed():
    # Problem 1: 1/2 pass = 0.5, Problem 2: 0/2 = 0 → macro avg = 0.25
    results = [{"t1": True, "t2": False}, {"t1": False, "t2": False}]
    assert average_test_pass_rate(results) == pytest.approx(0.25)


def test_pass_at_k_numerical_stability_large_n():
    # Should not overflow when n is moderately large.
    val = pass_at_k(n=200, c=100, k=10)
    assert 0.0 < val < 1.0
    assert not math.isnan(val)
