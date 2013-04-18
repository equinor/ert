#!/usr/bin/python
import sys

PYTHONPATH = sys.argv[1]
sys.argv.append( PYTHONPATH )

test_module = __import__(sys.argv[2])
test_module.ctest_run( sys.argv[3:] )
