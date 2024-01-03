from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional


def read(path: Path, default: Optional[str] = None) -> Optional[str]:
    return path.read_text().strip() if path.exists() else default


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("-f", action="store_true", required=True)
    ap.add_argument("-x", action="store_true", required=True)
    ap.add_argument("-F", required=True)
    ap.add_argument("jobs", nargs="*")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    assert args.F == "json", "Mock qstat must have -Fjson"

    jobs_path = Path(os.environ["PYTEST_TMP_PATH"]) / "mock_jobs"
    jobs_output = {}
    for job in args.jobs:
        name = read(jobs_path / f"{job}.name")
        assert name is not None

        pid = read(jobs_path / f"{job}.pid")
        returncode = read(jobs_path / f"{job}.returncode")

        state = "Q"
        if returncode is not None:
            state = "F"
        elif pid is not None:
            state = "R"

        info: Dict[str, Any] = {
            "Job_Name": name,
            "job_state": state,
        }

        if returncode is not None:
            info["Exit_status"] = int(returncode)

        jobs_output.update({job: info})

    print(json.dumps({"Jobs": jobs_output}))


if __name__ == "__main__":
    main()
