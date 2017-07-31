#!/usr/bin/env python
import sys
import time 
from ert.enkf import EnKFMain, ResConfig


# This will instantiate the EnkFMain object and create a handle to
# "everything" ert related for this instance.
res_config = ResConfig( sys.argv[1] )
ert = EnKFMain( res_config )
site_config = ert.siteConfig( )

jobs = site_config.get_installed_jobs( )
for job in jobs:
    print job.name()
    print "   config    : %s" % job.get_config_file()
    print "   executable: %s" % job.get_executable( ) 
    print
