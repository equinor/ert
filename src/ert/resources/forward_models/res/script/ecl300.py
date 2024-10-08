#!/usr/bin/env python
import sys

from ecl_config import Ecl300Config
from ecl_run import run

if __name__ == "__main__":
    config = Ecl300Config()
    run(config, [arg for arg in sys.argv[1:] if len(arg) > 0])
