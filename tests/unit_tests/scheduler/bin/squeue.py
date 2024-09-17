"""
This script partially mocks the Slurm provided utility squeue:

"view information about jobs located in the Slurm scheduling queue"

"""

import argparse
import glob
import os
from pathlib import Path
from typing import Literal, Optional

JobState = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--format", type=str, default="%i %T")
    parser.add_argument("-h", "--noheader", action="store_true")
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument(
        "--job",
        type=str,
    )
    parser.add_argument("-w", action="store_true")
    return parser


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    assert args.noheader, "Mocked squeue requires noheader"
    assert (
        args.format == "%i %T"
    ), "Sorry, mocked squeue only supports one custom format."

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    for pidfile in glob.glob(f"{jobs_path}/*.pid"):
        job = pidfile.split("/")[-1].split(".")[0]
        pid = read(Path(pidfile))
        returncode = read(jobs_path / f"{job}.returncode")

        state: JobState = "PENDING"

        if pid is not None and returncode is None:
            state = "RUNNING"
        elif pid is not None and returncode is not None:
            if returncode == "0":
                state = "COMPLETED"
            if returncode != "0":
                state = "FAILED"

        if state in ["PENDING", "RUNNING"]:
            print(f"{job} {state}")


if __name__ == "__main__":
    main()
