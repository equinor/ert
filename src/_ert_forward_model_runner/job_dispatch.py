import os
import signal
import sys

from _ert_forward_model_runner.cli import main as job_runner_main


def sigterm_handler(_signo, _stack_frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(0, signal.SIGTERM)


def main():
    os.nice(19)
    signal.signal(signal.SIGTERM, sigterm_handler)
    job_runner_main(sys.argv)


if __name__ == "__main__":
    main()
