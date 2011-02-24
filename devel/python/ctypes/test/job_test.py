#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'job_test.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


import os
import time
import sys
import shutil
import os.path
import ert.ecl       as ecl
import ert.job_queue as job_queue
import socket

server_list = { "be" : "lsf-be.no",
                "st" : "lsf-st.no",
                "tr" : "lsf-tr.no" }

src_files     = ["data/eclipse/case/ECLIPSE.DATA" , "data/eclipse/case/include"]
run_path_fmt  = "tmp/simulations/run%d"

default_driver_string = "LSF"
num_jobs              = 10
max_running           = 3


def copy_case( target_path , src_files):
    if not os.path.exists( target_path ):
        os.makedirs( target_path )

    print "Creating simulation directory:%s" % target_path
    for file in src_files:
        if os.path.isfile( file ):
            shutil.copy( file , target_path )
        elif os.path.isdir( file ):
            (path , base) = os.path.split( file )
            if not os.path.exists( "%s/%s" % (target_path , base)):
                shutil.copytree( file , "%s/%s" % (target_path , base) )
        else:
            sys.exit("Error")
            


def get_lsf_server():
    host = socket.gethostname()
    site = host[:2]
    lsf_server = server_list.get( site , False)
    if not lsf_server:
        print "Sorry - don't know what is the LSF server of in:%s" %  site
        sys.exit()
    return lsf_server

#################################################################

if len(sys.argv) == 1:
    driver_string = default_driver_string
else:
    driver_string = sys.argv[1].upper()

if driver_string == "LSF":
    driver = job_queue.LSFDriver( lsf_server = None )
elif driver_string == "LOCAL":
    driver = job_queue.LocalDriver()
else:
    sys.exit("Unrecognized driver string:%s" % driver_string)

queue = job_queue.EclQueue( driver , num_jobs , max_running )
queue.start()
case_list = []    
for case_nr in range( num_jobs ):
    copy_case( run_path_fmt % case_nr , src_files )
    case = ecl.EclCase( run_path_fmt % case_nr + "/ECLIPSE.DATA" )
    case.submit( queue )


while queue.running:                      
    print "Still running"
    time.sleep( 3 )



