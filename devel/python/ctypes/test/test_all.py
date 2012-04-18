#!/usr/bin/env python
import unittest
import sys

import sum_test
import sched_test
import large_mem_test


def run(name , suite):
    print "Running tests from %12s:" % name,
    sys.stdout.flush()
    unittest.TextTestRunner().run( suite )

    
run("summary"    , sum_test.suite())
run("sched_file" , sched_test.suite())
run("large_mem"  , large_mem_test.suite())
