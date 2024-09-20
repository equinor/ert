import argparse
import os
import time
from pathlib import Path
from textwrap import dedent
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mocked LSF bhist command reading state from filesystem"
    )
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("-l", action="store_true")
    parser.add_argument("jobs", type=str, nargs="*")
    return parser


def bhist_formatter(jobstats: List[Job]) -> str:
    string = "Summary of time in seconds spent in various states:\n"
    string += "JOBID   USER    JOB_NAME  PEND    PSUSP   RUN     USUSP   SSUSP   UNKWN   TOTAL\n"
    for job in jobstats:
        string += (
            f"{job.job_id:7.7s} {job.user:7.7s} "
            f"{job.job_name:9.9s} {job.pend!s:7.7s} "
            f"{job.psusp!s:7.7s} {job.run!s:7.7s} "
            f"{job.ususp!s:7.7s} {job.ssusp!s:7.7s} "
            f"{job.unkwn!s:7.7s} {job.total!s:7.7s}\n"
        )
    return string


def bhist_long_formatter(jobstats: List[Job]) -> str:
    """
    This function outputs stub data entirely independent from the input.
    """
    formatted_job_outputs = []
    for job in jobstats:
        job_output = dedent(
            f"""
            Job <{job.job_id}>, Job Name <{job.job_name}>, User <{job.user}>,
                                Project <default>, Command <sleep 100>
            Mon Apr 19 11:32:57: Submitted from host <test_host>, to Queue <normal>, CWD <$
                                HOME>;
            Mon Apr 19 11:32:57: Dispatched to <test_cluster>, Effective RES_REQ <select[
                                (cs)&&(type == any )&&(mem>maxmem*1/12)] span[hosts=1] >;
            Mon Apr 19 11:32:57: Starting (Pid 11111);
            Mon Apr 19 11:32:58: Running with execution home </private/test_user>, Execution CW
                                D </private/test_user>, Execution Pid <11111>;

            Summary of time in seconds spent in various states by  Mon Apr 19 11:33:14
            PEND     PSUSP    RUN      USUSP    SSUSP    UNKWN    TOTAL\n\t\t"""
            f"{job.pend!s:8.8s} {job.psusp!s:8.8s} {job.run!s:8.8s}"
            f"{job.ususp!s:8.8s} {job.ssusp!s:8.8s} {job.unkwn!s:8.8s}"
            f"{job.total!s:8.8s}"
        )
        formatted_job_outputs.append(job_output)
    return f"{50*'-'}".join(formatted_job_outputs)


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
            end_time_millis = int(
                os.path.getctime(jobs_path / f"{job}.returncode") * 1000
            )
            if not args.l:
                print("bhist says job is done")
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
    print(bhist_long_formatter(jobs_output) if args.l else bhist_formatter(jobs_output))


if __name__ == "__main__":
    main()
