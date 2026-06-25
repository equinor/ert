from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunResult:
    run_index: int
    elapsed_seconds: float
    exit_code: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ERT GUI launch+shutdown time across repeated runs, "
            "comparing cached starts to cold starts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ERT config file (e.g. test-data/ert/poly_example/poly.ert).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per mode (default: 10).",
    )
    parser.add_argument(
        "--mode",
        choices=("cached", "cold", "both"),
        default="both",
        help="Which benchmark mode to run (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchamark_data"),
        help="Directory where CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--cache-root",
        action="append",
        default=["src"],
        help=(
            "Repository-relative directory where __pycache__ folders are removed "
            "before each cold run. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help=(
            "On macOS, detect when the GUI window appears and close it "
            "programmatically (no manual close needed)."
        ),
    )
    parser.add_argument(
        "--auto-close-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the first window before fallback terminate.",
    )
    parser.add_argument(
        "--auto-close-poll-interval",
        type=float,
        default=0.05,
        help="Polling interval in seconds while waiting for the first window.",
    )
    return parser.parse_args()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def _summarize(results: list[RunResult]) -> dict[str, float | int]:
    timings = sorted(result.elapsed_seconds for result in results)
    return {
        "n": len(timings),
        "min": min(timings),
        "p50": statistics.median(timings),
        "mean": statistics.mean(timings),
        "p90": _percentile(timings, 0.9),
        "max": max(timings),
    }


def _clear_pycache(repo_root: Path, cache_roots: list[str]) -> int:
    removed = 0
    for rel_root in cache_roots:
        root = repo_root / rel_root
        if not root.exists():
            continue
        for pycache_dir in root.rglob("__pycache__"):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                removed += 1
    return removed


def _wait_for_window_when_visible_macos(
    pid: int, timeout_seconds: float, poll_interval: float
) -> bool:
    script = r'''
on run argv
    set targetPid to (item 1 of argv) as integer
    set timeoutSeconds to (item 2 of argv) as real
    set pollInterval to (item 3 of argv) as real
    set maxIters to timeoutSeconds / pollInterval

    tell application "System Events"
        repeat with i from 1 to maxIters
            set targetProcesses to (every process whose unix id is targetPid)
            if (count of targetProcesses) > 0 then
                set targetProcess to item 1 of targetProcesses
                if (count of windows of targetProcess) > 0 then
                    return "visible"
                end if
            end if
            delay pollInterval
        end repeat
    end tell

    return "timeout"
end run
'''

    completed = subprocess.run(
        [
            "osascript",
            "-e",
            script,
            str(pid),
            str(timeout_seconds),
            str(poll_interval),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return False
    return completed.stdout.strip() == "visible"


def _run_and_optionally_auto_close(
    *,
    base_cmd: list[str],
    repo_root: Path,
    env: dict[str, str],
    auto_close: bool,
    auto_close_timeout: float,
    auto_close_poll_interval: float,
) -> int:
    if not auto_close:
        return subprocess.run(base_cmd, cwd=repo_root, env=env, check=False).returncode

    process = subprocess.Popen(
        base_cmd,
        cwd=repo_root,
        env=env,
        start_new_session=True,
    )

    window_visible = False
    if platform.system() == "Darwin":
        window_visible = _wait_for_window_when_visible_macos(
            pid=process.pid,
            timeout_seconds=auto_close_timeout,
            poll_interval=auto_close_poll_interval,
        )

    if auto_close and platform.system() == "Darwin" and not window_visible:
        print(
            "[auto-close] Could not detect a visible window via System Events. "
            "Grant Accessibility permission to Terminal/VS Code in "
            "System Settings > Privacy & Security > Accessibility."
        )

    if auto_close:
        print(f"[auto-close] Terminating GUI process (pid={process.pid}) ")
        if platform.system() == "Windows":
            process.terminate()
        else:
            os.killpg(process.pid, 9)
    return process.wait()


def _run_mode(
    mode: str,
    runs: int,
    repo_root: Path,
    config: Path,
    cache_roots: list[str],
    auto_close: bool,
    auto_close_timeout: float,
    auto_close_poll_interval: float,
) -> list[RunResult]:
    results: list[RunResult] = []
    base_cmd = [sys.executable, "-m", "ert", "gui", str(config)]

    print(f"\nStarting mode: {mode} ({runs} runs)")
    for run_index in range(1, runs + 1):
        if mode == "cold":
            removed = _clear_pycache(repo_root=repo_root, cache_roots=cache_roots)
            print(f"[cold {run_index}/{runs}] removed __pycache__ dirs: {removed}")
        
        print(f"[{mode} {run_index}/{runs}] launching GUI")

        env = os.environ.copy()
        if mode == "cold":
            env["PYTHONDONTWRITEBYTECODE"] = "1"

        start = time.perf_counter()
        exit_code = _run_and_optionally_auto_close(
            base_cmd=base_cmd,
            repo_root=repo_root,
            env=env,
            auto_close=auto_close,
            auto_close_timeout=auto_close_timeout,
            auto_close_poll_interval=auto_close_poll_interval,
        )
        elapsed = time.perf_counter() - start

        results.append(
            RunResult(
                run_index=run_index,
                elapsed_seconds=elapsed,
                exit_code=exit_code,
            )
        )

        print(
            f"[{mode} {run_index}/{runs}] elapsed={elapsed:.3f}s "
            f"exit_code={exit_code}"
        )

    return results


def _write_outputs(
    output_dir: Path,
    timestamp: str,
    mode_results: dict[str, list[RunResult]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"gui_startup_{timestamp}.csv"
    json_path = output_dir / f"gui_startup_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["mode", "run_index", "elapsed_seconds", "exit_code"])
        for mode, results in mode_results.items():
            for result in results:
                writer.writerow(
                    [mode, result.run_index, result.elapsed_seconds, result.exit_code]
                )

    payload: dict[str, Any] = {"timestamp": timestamp, "modes": {}}
    for mode, results in mode_results.items():
        payload["modes"][mode] = {
            "summary": _summarize(results),
            "runs": [
                {
                    "run_index": result.run_index,
                    "elapsed_seconds": result.elapsed_seconds,
                    "exit_code": result.exit_code,
                }
                for result in results
            ],
        }

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    return csv_path, json_path


def _print_summary(mode_results: dict[str, list[RunResult]]) -> None:
    print("\nSummary")
    print("mode      n     min     p50    mean     p90     max")
    print("-----------------------------------------------------")

    summaries: dict[str, dict[str, float | int]] = {}
    for mode, results in mode_results.items():
        summary = _summarize(results)
        summaries[mode] = summary
        print(
            f"{mode:<8} {summary['n']:>2}  {summary['min']:>6.3f}  "
            f"{summary['p50']:>6.3f}  {summary['mean']:>6.3f}  "
            f"{summary['p90']:>6.3f}  {summary['max']:>6.3f}"
        )

    if "cold" in summaries and "cached" in summaries:
        cold_p50 = float(summaries["cold"]["p50"])
        cached_p50 = float(summaries["cached"]["p50"])
        print(f"\nDelta p50 (cold - cached): {cold_p50 - cached_p50:.3f}s")


def main() -> int:
    args = _parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    config = args.config
    if not config.is_absolute():
        config = (repo_root / config).resolve()

    if not config.is_file():
        raise FileNotFoundError(f"Config file not found: {config}")

    modes = [args.mode] if args.mode in {"cached", "cold"} else ["cached", "cold"]
    mode_results: dict[str, list[RunResult]] = {}

    for mode in modes:
        mode_results[mode] = _run_mode(
            mode=mode,
            runs=args.runs,
            repo_root=repo_root,
            config=config,
            cache_roots=args.cache_root,
            auto_close=args.auto_close,
            auto_close_timeout=args.auto_close_timeout,
            auto_close_poll_interval=args.auto_close_poll_interval,
        )

    _print_summary(mode_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path = _write_outputs(
        output_dir=args.output_dir,
        timestamp=timestamp,
        mode_results=mode_results,
    )
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
