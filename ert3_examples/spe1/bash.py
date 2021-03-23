#!/usr/bin/env python

# TODO: This script is a hack in its entirety to run shell commands. The reason
# is that the prefect evaluator and hence ert3 currently runs all commands
# within the unix step with "python3 cmd <args>". Over time this script should
# disappear...

import subprocess
import sys

cmd = list(sys.argv[1:])
subprocess.run(" ".join(cmd), shell=True, check=True)
