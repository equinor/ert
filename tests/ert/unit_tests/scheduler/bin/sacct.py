"""
This script partially mocks the Slurm provided utility sacct:

"view or modify Slurm configuration and state"

"""

import argparse
import os
from pathlib import Path
from typing import Literal

JobState = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-X", action="store_true")
    parser.add_argument("-n", action="store_true")
    parser.add_argument("-o", type=str)
    parser.add_argument("-P", action="store_true")
    parser.add_argument("-j", type=str)
    return parser


def read(path: Path, default: str | None = None) -> str | None:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    assert args.o.strip() == "State,ExitCode"

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    for job_dir in jobs_path.iterdir():
        pidfile = job_dir / "pid"
        job = job_dir.name
        if args.j and job != args.j:
            continue
        pid = read(pidfile)
        returncode = read(job_dir / "returncode")
        cancelled = read(job_dir / "cancelled", default="no")
        state: JobState = "PENDING"

        if pid is not None and returncode is None:
            state = "RUNNING"
        elif pid is not None and cancelled == "yes":
            state = "CANCELLED"
            returncode = 0
        elif pid is not None and returncode is not None:
            if returncode == "0":
                state = "COMPLETED"
            if returncode != "0":
                state = "FAILED"

        print(f"{state}|{returncode}:0")


if __name__ == "__main__":
    main()
