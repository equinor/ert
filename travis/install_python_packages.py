#!/usr/bin/env python

import sys
import subprocess

if (sys.version_info.major == 2):
   print("Installing Python 2.7 specific packages")
   subprocess.call(["conda", "install", "pyqt=4", "scipy=0.16.1", "future", "pylint", "numpy", "matplotlib", "pandas"])
elif (sys.version_info.major == 3):
   print("Installing Python 3 specific packages")
   subprocess.call(["conda", "install", "pyqt=5", "scipy=0.19.0", "future", "pylint", "numpy", "matplotlib", "pandas"])
