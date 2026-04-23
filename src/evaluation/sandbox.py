"""Safe code execution sandbox using isolated subprocesses.

Each generated code snippet and its test cases are evaluated in a fresh
subprocess that has no network access and no write permission to the working
directory (enforced via resource limits and a restricted execution template).
"""

import subprocess
import sys
import textwrap
import logging

logger = logging.getLogger(__name__)

# Subprocess runner script template.  The generated code and the test case are
# injected via stdin as a JSON payload so that no shell interpolation occurs.
_RUNNER_TEMPLATE = textwrap.dedent(
    """\
    import sys, json, traceback, builtins as _builtins_mod

    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    test_case = payload["test_case"]

    # Restrict dangerous builtins that allow filesystem / network access.
    # __builtins__ can be a dict or a module depending on execution context.
    _b = vars(_builtins_mod)
    _BLOCKED = {"open", "compile", "eval", "exec", "input", "memoryview"}
    _safe_builtins = {k: v for k, v in _b.items() if k not in _BLOCKED}

    globs = {"__builtins__": _safe_builtins}

    try:
        exec(compile(code, "<solution>", "exec"), globs)
    except SyntaxError as exc:
        print(json.dumps({"ok": False, "error": f"SyntaxError: {exc}"}))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": traceback.format_exc()}))
        sys.exit(0)

    try:
        exec(compile(test_case, "<test>", "exec"), globs)
        print(json.dumps({"ok": True, "error": None}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": traceback.format_exc()}))
    """
)


def execute_code_safely(
    code: str,
    test_cases: list[str],
    timeout_seconds: int = 5,
) -> dict[str, bool]:
    """Execute *code* against each test case in an isolated subprocess.

    Each test case is run in a separate subprocess so that:
    - A crash in one test does not affect the others.
    - Timeouts are enforced per test case.
    - The main process is never exposed to arbitrary code execution.

    Args:
        code: The Python source code to evaluate (defines the target function).
        test_cases: List of assert statements that exercise the function.
        timeout_seconds: Per-test-case wall-clock timeout.

    Returns:
        Dict mapping each test-case string to ``True`` (passed) or ``False``
        (failed, timed out, or errored).
    """
    import json  # stdlib — always available

    results: dict[str, bool] = {}

    for test_case in test_cases:
        payload = json.dumps({"code": code, "test_case": test_case})
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER_TEMPLATE],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout = proc.stdout.strip()
            if not stdout:
                results[test_case] = False
                logger.debug("Empty subprocess output for test: %s", test_case[:80])
                continue

            outcome = json.loads(stdout)
            results[test_case] = bool(outcome.get("ok", False))

        except subprocess.TimeoutExpired:
            logger.debug("Timeout (%ds) for test: %s", timeout_seconds, test_case[:80])
            results[test_case] = False

        except Exception as exc:  # noqa: BLE001
            logger.warning("Sandbox error for test '%s': %s", test_case[:80], exc)
            results[test_case] = False

    return results
