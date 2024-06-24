"""
This script partially mocks the Slurm provided utility scontrol:

"view or modify Slurm configuration and state"

"""

import argparse
import glob
import os
from pathlib import Path
from typing import Literal, Optional

JobState = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("format")
    parser.add_argument("jobstr")
    parser.add_argument("jobid", type=str, default="")
    return parser


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    for pidfile in glob.glob(f"{jobs_path}/*.pid"):
        job = pidfile.split("/")[-1].split(".")[0]
        if args.jobid and job != args.jobid:
            continue
        pid = read(Path(pidfile))
        returncode = read(jobs_path / f"{job}.returncode")
        name = read(jobs_path / f"{job}.name")
        cancelled = read(jobs_path / f"{job}.cancelled", default="no")
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

        print(f"JobId={job} JobName={name}")
        print(f"   JobState={state}")
        if returncode:
            print(f"   ExitCode={returncode}:0")
        print("")


if __name__ == "__main__":
    main()
