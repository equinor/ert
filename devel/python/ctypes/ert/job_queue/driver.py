#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'driver.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import time
import datetime
import ctypes
import sys
import os
import libjob_queue
import ert.util.SDP     as     SDP
from   job              import Job 
from   ert.cwrap.cwrap  import *
import ert.ecl.ecl_util    as ecl_util  
import ert.ecl.ecl_default as ecl_default  


LSF_TYPE   = 1
LOCAL_TYPE = 2
RSH_TYPE   = 3 



def init_LSF_env():
    os.environ["LSF_BINDIR"]    = "%s/bin" % libjob_queue.LSF_HOME
    os.environ["LSF_LIBDIR"]    = "%s/lib" % libjob_queue.LSF_HOME
    os.environ["XLSF_UIDDIR"]   = "%s/lib/uid" % libjob_queue.LSF_HOME
    os.environ["LSF_SERVERDIR"] = "%s/etc" % libjob_queue.LSF_HOME
    os.environ["LSF_ENVDIR"]    = "/prog/LSF/conf"

    os.environ["PATH"] += ":%s" % os.environ["LSF_BINDIR"]
    


class Driver:
    
    def is_driver_instance( self ):
        return True

    def from_param(self):
        return ctypes.c_void_p( self.c_ptr )

    def __del__( self ):
        cfunc.free_driver( self )

    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = cfunc.submit( self , cmd , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )
        if blocking:
            job.block()
            job = None
        
        return job
        

    def submit_ecl( self , data_file , eclipse_cmd = ecl_default.cmd , eclipse_version = ecl_default.version , blocking = False):
        (path_base , ext) = os.path.splitext( data_file )
        (run_path , base) = os.path.split( path_base )
        num_cpu = "%s" % ecl_util.get_num_cpu( data_file )
        return cfunc.submit( base , eclipse_cmd , run_path , [ eclipse_version , data_file , num_cpu ], blocking = blocking)
    

    def free_job( self , job ):
        cfunc.free_job( job )
    
        
    def get_status( self , job ):
        return cfunc.cget_status( self , job )
    

    def kill_job( self , job ):
        cfunc.ckill_job( self , job )
        


class LSFDriver(Driver):
    __initialized = False

    def __init__(self ,
                 lsf_server = None,
                 queue = "normal" ,
                 num_cpu = 1,
                 resource_request = ecl_default.lsf_resource_request):
        if not self.__initialized:
            init_LSF_env()
            LSFDriver.__initialized = True

        self.c_ptr = cfunc.alloc_driver_lsf( queue , resource_request , lsf_server , num_cpu)


class LocalDriver(Driver):

    def __init__( self ):
        self.c_ptr = cfunc.alloc_driver_local( )

RSH_HOST = "RSH_HOST"  # Value taken from rsh_driver.h
class RSHDriver(Driver):

    # Changing shell to bash can come in conflict with running ssh
    # commands.
    
    def __init__( self , rsh_host_list , rsh_cmd = "/usr/bin/ssh" ):
        """@rsh_host_list should be a list of tuples like: (hostname , max_running) 
        """

        self.c_ptr = cfunc.alloc_driver( rhs_cmd , None ) 
        
        for (host,max_running) in rsh_host_list:
            cfunc.set_str_option( self , RSH_HOST , "%s:%d" % (host , max_running))
            


#################################################################
cwrapper = CWrapper( libjob_queue.lib )
cwrapper.registerType( "driver" , Driver ) 
cwrapper.registerType( "job"    , Job )
cfunc   = CWrapperNameSpace( "driver" )

cfunc.alloc_driver_lsf       = cwrapper.prototype("c_void_p    queue_driver_alloc_LSF( char* , char* , char* , int )")
cfunc.alloc_driver_local     = cwrapper.prototype("c_void_p    queue_driver_alloc_local( )")
cfunc.alloc_driver_rsh       = cwrapper.prototype("c_void_p    queue_driver_alloc_RSH( char* , c_void_p )")


cfunc.free_driver    = cwrapper.prototype("void        queue_driver_free( driver )")
cfunc.submit         = cwrapper.prototype("c_void_p    queue_driver_submit_job( driver , char* , char* , char* , int , char**)")
cfunc.free_job       = cwrapper.prototype("void        queue_driver_free_job( job )")
cfunc.get_status     = cwrapper.prototype("int         queue_driver_get_status( driver , job)")
cfunc.kill_job       = cwrapper.prototype("void        queue_driver_kill_job( driver , job )")
cfunc.set_str_option = cwrapper.prototype("void        queue_driver_set_string_option( driver , int , char*)")
