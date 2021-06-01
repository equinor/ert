#!/usr/bin/env python

import os
import signal
import sys

from job_runner.cli import main


def sigterm_handler(_signo, _stack_frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(0, signal.SIGTERM)
#################################################################


#################################################################
os.nice(19)
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
    main(sys.argv)
