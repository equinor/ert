import time
import datetime
import ctypes
import sys
import os
import libjob_queue
from   ert.cwrap.cwrap import *

# Enum values nicked from libjob_queue/src/basic_queue_driver.h
STATUS_PENDING =  16
STATUS_RUNNING =  32
STATUS_DONE    =  64
STATUS_EXIT    = 128 

LSF_HOME    = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib"



class LSFDriver:

    def from_param(self):
        return self.c_ptr


    def __init__(self ,
                 queue = "normal" ,
                 resource_request = "select[cs && x86_64Linux] rusage[ecl100v2000=1:duration=5]" ,
                 lsf_server = None ):
        self.c_ptr = cfunc_lsf.alloc_driver( queue , resource_request , lsf_server , 1)


    def __del__( self ):
        cfunc_lsf.free_driver( self )


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = cfunc_lsf.submit( self , cmd , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )

        return job
        

    def free_job( self , job ):
        cfunc_lsf.free_job( job )
    

    def get_status( self , job ):
        return cfunc_lsf.get_status( self , job )
    

    def kill_job( self , job ):
        cfunc_lsf.kill_job( self , job )


        

class LocalDriver:
    def __init__( self ):
        self.c_ptr = cfunc_local.alloc_driver( )


    def __del__( self ):
        cfunc_local.free_driver( self )

    def from_param(self):
        return self.c_ptr


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        c_ptr = cfunc_local.submit( self , cmd , run_path , name , argc , argv )
        job = Job( self , c_ptr , blocking)
        return job
        

    def free_job( self , job ):
        cfunc_local.free_job( job )
    

    def get_status( self , job ):
        return cfunc_local.get_status( self , job )
    

    def kill_job( self , job ):
        cfunc_local.kill_job( self , job )
    
        
        

class Job:
    def __init__(self , driver , c_ptr , blocking = False):
        self.driver      = driver
        self.c_ptr       = c_ptr
        self.submit_time = datetime.datetime.now()
        if blocking:
            while True:
                status = self.status()
                if status == STATUS_DONE or status == STATUS_EXIT:
                    break
                else:
                    time.sleep( 2 )


    def __del__(self):
        self.driver.free_job( self )

    def run_time( self ):
        td = datetime.datetime.now() - self.submit_time
        return td.seconds + td.days * 24 * 3600

    def status( self ):
        return self.driver.get_status( self )
    
    def kill( self ):
        self.driver.kill_job( self )
    
    def from_param(self):
        return self.c_ptr


#################################################################
cwrapper = CWrapper( libjob_queue.lib )
cwrapper.registerType( "lsf_driver" , LSFDriver )
cwrapper.registerType( "local_driver" , LocalDriver )
cwrapper.registerType( "job" , Job )
cfunc_lsf   = CWrapperNameSpace( "driver" )
cfunc_local = CWrapperNameSpace( "driver" )

cfunc_lsf.alloc_driver = cwrapper.prototype("long    lsf_driver_alloc( char* , char* , char* , int )")
cfunc_lsf.free_driver  = cwrapper.prototype("void    lsf_driver_free( lsf_driver )")
cfunc_lsf.submit       = cwrapper.prototype("long    lsf_driver_submit_job( lsf_driver , char* , char* , char* , int , char**)")
cfunc_lsf.free_job     = cwrapper.prototype("void    lsf_driver_free_job( job )")
cfunc_lsf.get_status   = cwrapper.prototype("int     lsf_driver_get_job_status( lsf_driver , job)")
cfunc_lsf.kill_job     = cwrapper.prototype("void    lsf_driver_kill_job( lsf_driver , job )")


cfunc_local.alloc_driver     = cwrapper.prototype("long    local_driver_alloc( )")
cfunc_local.free_driver      = cwrapper.prototype("void    local_driver_free( local_driver )")
cfunc_local.submit           = cwrapper.prototype("long    local_driver_submit_job( local_driver , char* , char* , char* , int , char**)")
cfunc_local.free_job         = cwrapper.prototype("void    local_driver_free_job( job )")
cfunc_local.kill_job         = cwrapper.prototype("void    local_driver_kill_job( local_driver , job )")
cfunc_local.get_status       = cwrapper.prototype("int     local_driver_get_job_status( local_driver , job)")

