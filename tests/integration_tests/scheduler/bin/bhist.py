import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class Job(BaseModel):
    job_id: str
    user: str = "username"
    job_name: str = "d u m m y"  # can be up to 4094 chars
    pend: int = 0
    psusp: int = 0
    run: int = 0
    ususp: int = 0
    ssusp: int = 0
    unkwn: int = 0
    total: int = 0


class SQueueOutput(BaseModel):
    jobs: List[Job]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mocked LSF bhist command reading state from filesystem"
    )
    parser.add_argument("jobs", type=str, nargs="*")
    return parser


def bhist_formatter(jobstats: List[Job]) -> str:
    string = "Summary of time in seconds spent in various states:\n"
    string += "JOBID   USER    JOB_NAME  PEND    PSUSP   RUN     USUSP   SSUSP   UNKWN   TOTAL\n"
    for job in jobstats:
        string += (
            f"{str(job.job_id):7.7s} {job.user:7.7s} "
            f"{job.job_name:9.9s} {str(job.pend):7.7s} "
            f"{str(job.psusp):7.7s} {str(job.run):7.7s} "
            f"{str(job.ususp):7.7s} {str(job.ssusp):7.7s} "
            f"{str(job.unkwn):7.7s} {str(job.total):7.7s}\n"
        )
    return string


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def main() -> None:
    args = get_parser().parse_args()

    jobs_path = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    jobs_output: List[Job] = []
    for job in args.jobs:
        job_name: str = read(jobs_path / f"{job}.name") or "_"
        assert job_name is not None

        submit_time_millis: int = int(
            os.path.getctime(jobs_path / f"{job}.name") * 1000
        )
        pending_time_millis = int(read(jobs_path / "pendingtimemillis") or 0)
        run_start_time_millis: int = submit_time_millis + pending_time_millis
        end_time_millis: int = int(time.time() * 1000)
        if (jobs_path / f"{job}.returncode").exists():
            print("bhist says job is done")
            end_time_millis = int(
                os.path.getctime(jobs_path / f"{job}.returncode") * 1000
            )
            print(f"run: {end_time_millis - run_start_time_millis}")
        pend: int = min(
            run_start_time_millis - submit_time_millis,
            int(time.time() * 1000) - submit_time_millis,
        )

        jobs_output.append(
            Job(
                **{
                    "job_id": job,
                    "user": "dummyuser",
                    "job_name": job_name,
                    "pend": pend,
                    "run": max(0, end_time_millis - run_start_time_millis),
                    "total": end_time_millis - submit_time_millis,
                }
            )
        )
    print(bhist_formatter(jobs_output))


if __name__ == "__main__":
    main()
