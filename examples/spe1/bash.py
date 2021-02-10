#!/usr/bin/env python

import subprocess
import sys

cmd = list(sys.argv[1:])
subprocess.run(" ".join(cmd), shell=True, check=True)
