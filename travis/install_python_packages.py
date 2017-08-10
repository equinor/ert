#!/usr/bin/env python

import sys
from subprocess import call



if (sys.version_info.major == 2 and sys.version_info.minor == 7):
   call(["export", 'TRAVIS_PYTHON_VERSION="2.7"'])
   print("Setting python 2.7 env variable")
elif (sys.version_info.major == 3 and sys.version_info.minor == 6):
   call(["export", 'TRAVIS_PYTHON_VERSION="3.6"'])
   print("Setting python 3.6 env variable")
   
