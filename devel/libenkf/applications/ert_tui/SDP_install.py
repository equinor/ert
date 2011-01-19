#!/usr/bin/python
import sys
import commands
import os
sys.path += ["../../../python/ctypes/SDP"]
import SDP

local_ert   = "ert"
svn_version = commands.getoutput( "svnversion" )
svn_ert     = "%s_%s" % (local_ert , svn_version)
(SDP_ROOT , RH_version) = SDP.get_SDP_ROOT()
target_file = "%s/bin/ert_release/%s" % (SDP_ROOT, svn_ert)
ert_link    = "%s/bin/ert_latest_and_greatest" % SDP_ROOT

SDP.install_file( local_ert , target_file )
#print "Installed........: %s" % target_file

SDP.install_link( target_file , ert_link )
#print "Installed link...: %s -> %s" % ( ert_link , target_file )



