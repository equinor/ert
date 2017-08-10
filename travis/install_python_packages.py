#!/usr/bin/env python

import sys
import subprocess

call(["conda", "install", "future"])
call(["conda", "install", "pylint"])
call(["conda", "install", "numpy"])
call(["conda", "install", "matplotlib"])
call(["conda", "install", "pandas"])

if (sys.version_info.major == 2 and sys.version_info.minor == 7):
   print("Installing Python 2.7 specific packages")
   call(["conda", "install", "pyqt=4"])
   call(["conda", "install", "scipy=0.16.1"])
elif (sys.version_info.major == 3 and sys.version_info.minor == 6):
   print("Installing Python 3.6 specific packages")
   call(["conda", "install", "pyqt=5"])
   call(["conda", "install", "scipy=0.19.0"])

   
