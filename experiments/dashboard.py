#!/usr/bin/env python3
"""Live terminal dashboard for TFG MultiAgente experiments.

Usage:
    python experiments/dashboard.py

Reads progress.json and CSV files every 2 seconds. Run in a separate terminal
while run_experiments.py is executing.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

_RESULTS_DIR = Path("experiments/results")
_LOGS_DIR = Path("experiments/logs")
_PROGRESS_FILE = _RESULTS_DIR / "progress.json"

_CONFIG_KEYS = [
    "baseline",
    "sequential",
    "self_reflection_r1",
    "self_reflection_r2",
    "self_reflection_r3",
]

_CONFIG_CSV: dict[str, Path] = {
    k: _RESULTS_DIR / f"{k}_results.csv" for k in _CONFIG_KEYS
}

_TOTAL = 9100
_DASHBOARD_START = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_progress() -> dict:
    if not _PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def _compute_config_stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "runs_done": 0,
            "pass1": 0.0,
            "avg_tokens": 0.0,
            "avg_latency": 0.0,
        }

    try:
        from src.evaluation.metrics import pass_at_k  # type: ignore
    except ImportError:
        pass_at_k = None  # type: ignore

    by_problem: dict[str, dict] = {}
    total_tokens = 0
    total_latency = 0.0
    count = 0

    for row in rows:
        pid = row.get("problem_id", "?")
        if pid not in by_problem:
            by_problem[pid] = {"n": 0, "c": 0}
        by_problem[pid]["n"] += 1
        if str(row.get("pass_all_tests", "")).lower() in ("true", "1"):
            by_problem[pid]["c"] += 1

        try:
            total_tokens += int(row.get("tokens_input", 0)) + int(row.get("tokens_output", 0))
            total_latency += float(row.get("latency_seconds", 0))
            count += 1
        except (ValueError, TypeError):
            pass

    if pass_at_k is not None and by_problem:
        vals = [pass_at_k(d["n"], d["c"], 1) for d in by_problem.values()]
        pass1 = sum(vals) / len(vals)
    else:
        pass1 = 0.0

    return {
        "runs_done": len(rows),
        "pass1": pass1,
        "avg_tokens": total_tokens / count if count else 0.0,
        "avg_latency": total_latency / count if count else 0.0,
    }


def _all_recent_rows(limit: int = 8) -> list[dict]:
    all_rows: list[dict] = []
    for key, path in _CONFIG_CSV.items():
        for row in _read_csv(path):
            row["_config_key"] = key
            all_rows.append(row)

    # Sort by timestamp descending, fallback to row order
    def _ts(r: dict) -> str:
        return r.get("timestamp", "")

    all_rows.sort(key=_ts, reverse=True)
    return all_rows[:limit]


def _read_error_log(n: int = 3) -> list[str]:
    log_path = _LOGS_DIR / "errors.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        # Return last n non-empty lines
        non_empty = [l for l in lines if l.strip()]
        return non_empty[-n:]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_header() -> Panel:
    now = datetime.now(timezone.utc)
    elapsed = now - _DASHBOARD_START
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    content = Text.assemble(
        ("TFG MultiAgente — Experiment Dashboard", "bold cyan"),
        "   ",
        (now.strftime("%Y-%m-%d %H:%M:%S UTC"), "dim"),
        "   ",
        (f"Elapsed: {elapsed_str}", "dim"),
    )
    return Panel(content, style="bold")


def _render_overall(prog: dict) -> Panel:
    total = prog.get("total", _TOTAL) or _TOTAL
    completed = prog.get("completed", 0)
    failed = prog.get("failed", 0)
    remaining = max(0, total - completed)

    bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=50),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
    )
    task_id = bar.add_task("Overall", total=total, completed=completed)

    # ETA
    last_updated = prog.get("last_updated", "")
    eta_str = "—"
    if completed > 0 and last_updated:
        try:
            lu = datetime.fromisoformat(last_updated)
            elapsed_s = (_DASHBOARD_START - lu).total_seconds()
            if elapsed_s < 0:
                elapsed_s = (datetime.now(timezone.utc) - _DASHBOARD_START).total_seconds()
            rate = completed / max(elapsed_s, 1)
            eta_s = remaining / rate if rate > 0 else 0
            eta_m = int(eta_s // 60)
            eta_str = f"~{eta_m} min" if eta_m < 60 else f"~{eta_m//60}h {eta_m%60}m"
        except Exception:
            pass

    stats = Text.assemble(
        ("Completed: ", "dim"), (str(completed), "green bold"),
        ("  Failed: ", "dim"), (str(failed), "red bold"),
        ("  Remaining: ", "dim"), (str(remaining), "yellow bold"),
        ("  ETA: ", "dim"), (eta_str, "cyan"),
    )

    from rich.console import Group
    return Panel(Group(bar.get_renderable(), stats), title="Overall Progress")


def _render_config_table(all_stats: dict[str, dict]) -> Panel:
    runs_per_config = _TOTAL // len(_CONFIG_KEYS)

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Config", min_width=22)
    table.add_column("Done", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Pass@1", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Status")

    for key in _CONFIG_KEYS:
        stats = all_stats.get(key, {})
        done = stats.get("runs_done", 0)
        pass1 = stats.get("pass1", 0.0)
        avg_tok = stats.get("avg_tokens", 0.0)
        avg_lat = stats.get("avg_latency", 0.0)

        if done >= runs_per_config:
            status = "[green]✅ complete[/green]"
        elif done > 0:
            status = "[yellow]🔄 running[/yellow]"
        else:
            status = "[dim]⏳ pending[/dim]"

        table.add_row(
            key,
            str(done),
            str(runs_per_config),
            f"{pass1:.1%}",
            f"{avg_tok:,.0f}",
            f"{avg_lat:.1f}s",
            status,
        )

    return Panel(table, title="Per-Config Progress")


def _render_current(prog: dict) -> Panel:
    current = prog.get("current", {})
    if not current:
        content = Text("Waiting for experiments to start…", style="dim italic")
    else:
        content = Text.assemble(
            ("Benchmark: ", "dim"), (str(current.get("benchmark", "?")), "cyan bold"),
            ("  Problem: ", "dim"), (str(current.get("problem_id", "?")), "cyan bold"),
            ("  Config: ", "dim"), (str(current.get("config", "?")), "cyan bold"),
            ("  Seed: ", "dim"), (str(current.get("seed", "?")), "cyan bold"),
            "  ",
            ("⟳ running", "yellow bold"),
        )
    return Panel(content, title="Current Run")


def _render_recent(rows: list[dict]) -> Panel:
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Problem ID", min_width=16)
    table.add_column("Config", min_width=20)
    table.add_column("Seed", justify="right")
    table.add_column("Pass", justify="center")
    table.add_column("Rate", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency", justify="right")

    for row in rows:
        pass_all = str(row.get("pass_all_tests", "")).lower() in ("true", "1")
        rate_str = row.get("test_pass_rate", "0")
        try:
            rate = float(rate_str)
        except (ValueError, TypeError):
            rate = 0.0

        if pass_all:
            style = "green"
        elif rate > 0:
            style = "yellow"
        else:
            style = "red"

        try:
            tokens = int(row.get("tokens_input", 0)) + int(row.get("tokens_output", 0))
        except (ValueError, TypeError):
            tokens = 0
        try:
            latency = float(row.get("latency_seconds", 0))
        except (ValueError, TypeError):
            latency = 0.0

        table.add_row(
            row.get("problem_id", "?"),
            row.get("config", "?"),
            str(row.get("seed", "?")),
            "✓" if pass_all else "✗",
            f"{rate:.0%}",
            f"{tokens:,}",
            f"{latency:.1f}s",
            style=style,
        )

    return Panel(table, title="Recent Results (last 8)")


def _render_errors(error_lines: list[str]) -> Panel:
    if not error_lines:
        content = Text("No errors logged.", style="dim")
    else:
        content = Text("\n".join(error_lines), style="red dim", overflow="ellipsis")
    return Panel(content, title="Recent Errors")


def _build_display(prog: dict, all_stats: dict[str, dict]) -> object:
    from rich.console import Group

    recent = _all_recent_rows(8)
    error_lines = _read_error_log(3)

    if not prog:
        waiting = Panel(
            Text("Waiting for experiments to start…", style="dim italic", justify="center"),
            title="TFG MultiAgente — Experiment Dashboard",
            style="dim",
        )
        return waiting

    return Group(
        _render_header(),
        _render_overall(prog),
        _render_config_table(all_stats),
        _render_current(prog),
        _render_recent(recent),
        _render_errors(error_lines),
    )

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    console = Console()

    try:
        with Live(console=console, refresh_per_second=0.5, screen=False) as live:
            while True:
                prog = _read_progress()
                all_stats: dict[str, dict] = {}
                for key, path in _CONFIG_CSV.items():
                    rows = _read_csv(path)
                    all_stats[key] = _compute_config_stats(rows)

                live.update(_build_display(prog, all_stats))

                import time
                time.sleep(2)

    except KeyboardInterrupt:
        total = _read_progress().get("total", _TOTAL)
        completed = _read_progress().get("completed", 0)
        console.print(
            f"\n[bold cyan]Dashboard closed.[/bold cyan] "
            f"Progress: {completed}/{total} runs completed."
        )


if __name__ == "__main__":
    main()
