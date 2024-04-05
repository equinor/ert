from __future__ import annotations

import json
import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional

QSTAT_HEADER = (
    "Job id                         Name            User             Time Use S Queue\n"
    "-----------------------------  --------------- ---------------  -------- - ---------------\n"
)


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("-f", action="store_true")
    ap.add_argument("-x", action="store_true", required=True)
    ap.add_argument("-F", default="")
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("-w", action="store_true")
    ap.add_argument("-E", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.f:
        assert args.F == "json", "full format can only be outputted in json format"

    jobs_path = Path(os.environ.get("PYTEST_TMP_PATH", ".")) / "mock_jobs"

    jobs_output = {}

    assert isinstance(args.F, str)
    if not args.F:
        print(QSTAT_HEADER, end="")

    for job in args.jobs:
        name = read(jobs_path / f"{job}.name")
        assert name is not None

        pid = read(jobs_path / f"{job}.pid")
        returncode = read(jobs_path / f"{job}.returncode")

        state = "Q"
        if returncode is not None:
            state = random.choice("EF")
        elif pid is not None:
            state = "R"

        info: Dict[str, Any] = {
            "Job_Name": name,
            "job_state": state,
        }

        if returncode is not None:
            info["Exit_status"] = int(returncode)

        jobs_output.update({job: info})

        if not args.F:
            user = "mock"
            time = "00:00:00"
            queue = "mocked"

            print(
                f"{job:30.30}  {name:15.15} {user:15.15}  {time:8.8} {state:1.1} {queue:5.5}"
            )

    if args.F == "json":
        print(json.dumps({"Jobs": jobs_output}))


if __name__ == "__main__":
    main()
