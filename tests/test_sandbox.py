"""Tests for the subprocess sandbox.

The sandbox is the trust boundary between the LLM-generated code and the
host system. Its failure modes (false positives, hangs, escapes) directly
contaminate the pass@k metric, so it carries the most consequential
correctness guarantees in the project.
"""

from __future__ import annotations

from src.evaluation.sandbox import execute_code_safely


def test_passes_simple_function():
    code = "def add(a, b):\n    return a + b\n"
    results = execute_code_safely(code, ["assert add(2, 3) == 5"])
    assert results == {"assert add(2, 3) == 5": True}


def test_fails_wrong_implementation():
    code = "def add(a, b):\n    return a - b\n"
    results = execute_code_safely(code, ["assert add(2, 3) == 5"])
    assert results == {"assert add(2, 3) == 5": False}


def test_syntax_error_is_failure_not_crash():
    code = "def broken(:\n    pass\n"
    results = execute_code_safely(code, ["assert True"])
    assert results == {"assert True": False}


def test_runtime_error_is_failure():
    code = "def f():\n    return 1 / 0\n"
    results = execute_code_safely(code, ["assert f() == 0"])
    assert results == {"assert f() == 0": False}


def test_infinite_loop_times_out():
    code = "def loop():\n    while True:\n        pass\n"
    results = execute_code_safely(
        code, ["assert loop() is None"], timeout_seconds=2
    )
    assert results == {"assert loop() is None": False}


def test_multiple_tests_independent():
    code = "def f(x):\n    return x * 2\n"
    cases = [
        "assert f(2) == 4",
        "assert f(3) == 7",  # wrong on purpose
        "assert f(0) == 0",
    ]
    results = execute_code_safely(code, cases)
    assert results[cases[0]] is True
    assert results[cases[1]] is False
    assert results[cases[2]] is True


def test_blocked_builtin_open_fails():
    code = "def f():\n    return open('/etc/passwd').read()\n"
    results = execute_code_safely(code, ["assert f() is not None"])
    assert results == {"assert f() is not None": False}
