#!/usr/bin/env python
import sys
import os


PYTHONPATH = sys.argv[1]
test_module = sys.argv[2]
cwd = None
argv = []

sys.path.insert( 0 , PYTHONPATH )


test_module = __import__(sys.argv[2])

try:
    argv = sys.argv[3:]
except:
    pass

test_module.ctest_run( sys.argv[3:] )
