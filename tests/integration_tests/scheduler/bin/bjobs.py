import argparse
import datetime
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

JobState = Literal[
    "EXIT", "DONE", "PEND", "RUN", "ZOMBI", "PDONE", "SSUSP", "USUSP", "UNKWN"
]


class Job(BaseModel):
    job_id: str
    name: str
    job_state: JobState
    user_name: str = "username"
    queue: str = "normal"
    from_host: str = "localhost"
    exec_host: str = "localhost"
    submit_time: int = 0


class SQueueOutput(BaseModel):
    jobs: List[Job]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mocked LSF bjobs command reading state from filesystem"
    )
    parser.add_argument("jobs", type=str, nargs="*")
    return parser


def bjobs_formatter(jobstats: List[Job]) -> str:
    string = "JOBID USER     STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME\n"
    for job in jobstats:
        submit_time = datetime.datetime.fromtimestamp(
            job.submit_time, datetime.timezone.utc
        )
        string += (
            f"{str(job.job_id):<5s} {job.user_name:<8s} "
            f"{job.job_state:<4s} {job.queue:<8} "
            f"{job.from_host:<11s} {job.exec_host:<11s} {job.name:<8s} "
            f"{submit_time}\n"
        )
    return string


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    jobs_output: List[Job] = []
    for job in args.jobs:
        name: str = read(jobs_path / f"{job}.name") or "_"
        assert name is not None

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
                    "name": name,
                    "job_state": state,
                }
            )
        )

    print(bjobs_formatter(jobs_output))


if __name__ == "__main__":
    main()
