#!/usr/bin/env python

import sys
import subprocess

subprocess.call(["conda", "install", "future"])
subprocess.call(["conda", "install", "pylint"])
subprocess.call(["conda", "install", "numpy"])
subprocess.call(["conda", "install", "matplotlib"])
subprocess.call(["conda", "install", "pandas"])

if (sys.version_info.major == 2 and sys.version_info.minor == 7):
   print("Installing Python 2.7 specific packages")
   subprocess.call(["conda", "install", "pyqt=4"])
   subprocess.call(["conda", "install", "scipy=0.16.1"])
elif (sys.version_info.major == 3 and sys.version_info.minor == 6):
   print("Installing Python 3.6 specific packages")
   subprocess.call(["conda", "install", "pyqt=5"])
   subprocess.call(["conda", "install", "scipy=0.19.0"])

   
