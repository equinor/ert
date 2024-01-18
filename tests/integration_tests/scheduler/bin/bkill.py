import argparse
import os
import signal
import sys
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kill jobs")
    parser.add_argument("jobids", type=str, nargs="+")
    return parser


def main() -> None:
    args = get_parser().parse_args()

    jobdir = Path(os.getenv("PYTEST_TMP_PATH", ".")) / "mock_jobs"
    killsignal = signal.SIGTERM
    for jobid in args.jobids:
        pidfile = jobdir / f"{jobid}.pid"
        if not pidfile.exists():
            sys.exit(1)
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(pid, killsignal)
        pidfile.unlink()


if __name__ == "__main__":
    main()
