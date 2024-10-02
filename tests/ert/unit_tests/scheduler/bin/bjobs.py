import argparse
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

JobState = Literal[
    "EXIT", "DONE", "PEND", "RUN", "ZOMBI", "PDONE", "SSUSP", "USUSP", "UNKWN"
]


class Job(BaseModel):
    job_id: str
    job_state: JobState


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mocked LSF bjobs command reading state from filesystem"
    )
    parser.add_argument("-o", type=str, default=None)
    parser.add_argument("-noheader", action="store_true")
    parser.add_argument("jobs", type=str, nargs="*")
    parser.add_argument("-w", action="store_true")
    return parser


def bjobs_formatter(jobstats: List[Job]) -> str:
    return "".join([f"{job.job_id}^{job.job_state}^-\n" for job in jobstats])


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    # this is for the bjobs call looking for exit code
    if args.o.strip() == "exit_code":
        returncode = read(jobs_path / f"{args.jobs[0]}.returncode")
        print(returncode)
        return

    jobs_output: List[Job] = []
    for job in args.jobs:
        pid = read(jobs_path / f"{job}.pid")
        returncode = read(jobs_path / f"{job}.returncode")

        state: JobState = "PEND"

        if returncode is not None:
            state = "DONE" if int(returncode) == 0 else "EXIT"
        elif pid is not None:
            state = "RUN"

        jobs_output.append(
            Job(
                **{
                    "job_id": job,
                    "job_state": state,
                }
            )
        )

    print(bjobs_formatter(jobs_output))


if __name__ == "__main__":
    main()
