#!/usr/bin/env python

import sys
from subprocess import call

call(["conda", "install", "future"])
call(["conda", "install", "pylint"])
call(["conda", "install", "numpy"])
call(["conda", "install", "matplotlib"])
call(["conda", "install", "pandas"])

if (sys.version_info.major == 2 and sys.version.minor == 7):
   call(["conda", "install", "pyqt=4"])
   call(["conda", "install", "scipy=0.16.1"])
elif (sys.version_info.major == 3 and sys.version.minor == 6):
   call(["conda", "install", "pyqt=5"])
   call(["conda", "install", "scipy=0.19.0"])
   
