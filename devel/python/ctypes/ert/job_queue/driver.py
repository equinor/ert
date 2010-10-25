import time
import datetime
import ctypes
import sys
import os
import libjob_queue
import ert.util.SDP    as     SDP
from   ert.cwrap.cwrap import *
from   job             import Job 



LSF_TYPE   = 1
LOCAL_TYPE = 2




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
        return self.c_ptr


    def __init__(self , type):
        if type == LSF_TYPE:
            self.free         = cfunc_lsf.free_driver
            self.csubmit      = cfunc_lsf.submit
            self.cget_status  = cfunc_lsf.get_status
            self.free_driver  = cfunc_lsf.free_driver
            self.ckill_job    = cfunc_lsf.kill_job
            self.free_job     = cfunc_lsf.free_job
        elif type == LOCAL_TYPE:
            self.free         = cfunc_local.free_driver
            self.csubmit      = cfunc_local.submit
            self.free_driver  = cfunc_local.free_driver
            self.cget_status  = cfunc_local.get_status
            self.ckill_job    = cfunc_local.kill_job
            self.free_job     = cfunc_local.free_job
        else:
            sys.exit("Error")
        
        
    def __del__( self ):
        self.free_driver( self )


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = self.csubmit( self , cmd , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )
        if blocking:
            pass
        
        return job
        

    def free_job( self , job ):
        self.free_job( job )
    
        
    def get_status( self , job ):
        return self.cget_status( self , job )
    

    def kill_job( self , job ):
        self.ckill_job( self , job )
        



class LSFDriver(Driver):
    __initialized = False

    def __init__(self ,
                 lsf_server = None,
                 queue = "normal" ,
                 resource_request = "select[cs && x86_64Linux] rusage[ecl100v2000=1:duration=5]"):
        Driver.__init__( self , LSF_TYPE )
        if not self.__initialized:
            init_LSF_env()
            LSFDriver.__initialized = True

        self.c_ptr = cfunc_lsf.alloc_driver( queue , resource_request , lsf_server , 1)


class LocalDriver(Driver):

    def __init__( self ):
        Driver.__init__( self , LOCAL_TYPE )
        self.c_ptr = cfunc_local.alloc_driver( )
        


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

