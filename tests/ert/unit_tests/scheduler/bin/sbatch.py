"""
This script partially mocks the Slurm provided utility sbatch:

"Submit a batch script to Slurm"
"""

import argparse
import os
import random
import subprocess
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, default="dummy")
    parser.add_argument("--chdir", type=str)
    parser.add_argument("--ntasks", type=int)
    parser.add_argument("--partition", type=str)
    parser.add_argument("--parsable", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--error", type=str)
    parser.add_argument("--mem", type=str)
    parser.add_argument("--nodelist", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--time", type=str)
    parser.add_argument("--account", type=str)
    parser.add_argument("script", type=str)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    jobid = random.randint(1, 2**15)
    jobsdir = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"
    jobdir = jobsdir / str(jobid)
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "script").write_text(args.script, encoding="utf-8")
    (jobdir / "name").write_text(args.job_name, encoding="utf-8")
    env_file = jobdir / "env"
    if args.ntasks:
        env_file.write_text(
            f"export SLURM_JOB_CPUS_PER_NODE={args.ntasks}\n"
            f"export SLURM_CPUS_ON_NODE={args.ntasks}",
            encoding="utf-8",
        )
    else:
        env_file.touch()

    subprocess.Popen(
        [str(Path(__file__).parent / "runner"), f"{jobdir}"],
        start_new_session=True,
        stdout=open(args.output, "w", encoding="utf-8"),  # noqa: SIM115
        stderr=open(args.error, "w", encoding="utf-8"),  # noqa: SIM115
    )

    if args.parsable:
        print(jobid)
    else:
        print(f"Submitted {jobid} to slurm")


if __name__ == "__main__":
    main()
