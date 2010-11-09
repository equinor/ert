#!/prog/sdpsoft/python2.4/bin/python
import sys

print  "import ert", 
sys.stdout.flush()
import ert
print

print "import ert.ecl",
sys.stdout.flush()
import ert.ecl
print

print "import ert.job_queue",
sys.stdout.flush()
import ert.job_queue 
print

print "import ert.util",
sys.stdout.flush()
import ert.util.stringlist
import ert.util.tvector
import ert.util.stat
print
