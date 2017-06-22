#!/usr/bin/env python
import sys
fileH = open(sys.argv[1] , "w")
for arg in sys.argv[2:]:
    fileH.write("%s\n" % arg)
fileH.close()


