#!/usr/bin/env python

import sys
import subprocess

if (sys.version_info.major == 2):
   print("Installing Python 2.7 specific packages")
   subprocess.call(["conda", "install", "pyqt=4"])
   subprocess.call(["conda", "install", "scipy=0.16.1"])
elif (sys.version_info.major == 3):
   print("Installing Python 3 specific packages")
   subprocess.call(["conda", "install", "pyqt=5"])
   subprocess.call(["conda", "install", "scipy=0.19.0"])

subprocess.call(["conda", "install", "future", "pylint", "numpy", "matplotlib", "pandas"])
