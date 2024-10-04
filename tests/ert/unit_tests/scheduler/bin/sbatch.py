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
    parser.add_argument("script", type=str)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    jobid = random.randint(1, 2**15)
    jobdir = Path(os.getenv("PYTEST_TMP_PATH", "."))
    (jobdir / "mock_jobs").mkdir(parents=True, exist_ok=True)
    (jobdir / "mock_jobs" / f"{jobid}.script").write_text(args.script, encoding="utf-8")
    (jobdir / "mock_jobs" / f"{jobid}.name").write_text(args.job_name, encoding="utf-8")
    env_file = jobdir / "mock_jobs" / f"{jobid}.env"

    if args.ntasks:
        env_file.write_text(
            f"export SLURM_JOB_CPUS_PER_NODE={args.ntasks}\n"
            f"export SLURM_CPUS_ON_NODE={args.ntasks}",
            encoding="utf-8",
        )
    else:
        env_file.touch()

    subprocess.Popen(
        [str(Path(__file__).parent / "runner"), f"{jobdir}/mock_jobs/{jobid}"],
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
