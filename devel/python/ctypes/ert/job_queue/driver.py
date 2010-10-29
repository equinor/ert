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
        elif type == RSH_TYPE:
            self.free         = cfunc_rsh.free_driver
            self.csubmit      = cfunc_rsh.submit
            self.free_driver  = cfunc_rsh.free_driver
            self.cget_status  = cfunc_rsh.get_status
            self.ckill_job    = cfunc_rsh.kill_job
            self.free_job     = cfunc_rsh.free_job
        else:
            sys.exit()
            
            
        
    def __del__( self ):
        self.free_driver( self )


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = self.csubmit( self , cmd , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )
        if blocking:
            job.block()
            job = None
        
        return job
        

    def submit_ecl( self , data_file , eclipse_cmd = ecl_default.cmd , eclipse_version = ecl_default.version , blocking = False):
        (path_base , ext) = os.path.splitext( data_file )
        (run_path , base) = os.path.split( path_base )
        num_cpu = "%s" % ecl_util.get_num_cpu( data_file )
        return self.submit( base , eclipse_cmd , run_path , [ eclipse_version , num_cpu ], blocking = blocking)
    

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
                 resource_request = ecl_default.lsf_resource_request):
        Driver.__init__( self , LSF_TYPE )
        if not self.__initialized:
            init_LSF_env()
            LSFDriver.__initialized = True

        self.c_ptr = cfunc_lsf.alloc_driver( queue , resource_request , lsf_server , 1)


class LocalDriver(Driver):

    def __init__( self ):
        Driver.__init__( self , LOCAL_TYPE )
        self.c_ptr = cfunc_local.alloc_driver( )


class RSHDriver(Driver):

    # Changing shell to bash can come in conflict with running ssh
    # commands.

    def __init__( self , rsh_host_list):
        """@rsh_host_list should be a list of tuples like: (hostname , max_running) 
        """

        Driver.__init__( self , RSH_TYPE )
        self.c_ptr = cfunc_rsh.alloc_driver( "/usr/bin/ssh" , None )
        
        for (host,max_running) in rsh_host_list:
            cfunc_rsh.add_host( self , host , max_running)
            


#################################################################
cwrapper = CWrapper( libjob_queue.lib )
cwrapper.registerType( "lsf_driver" , LSFDriver )
cwrapper.registerType( "local_driver" , LocalDriver )
cwrapper.registerType( "rsh_driver" , RSHDriver )
cwrapper.registerType( "job" , Job )
cfunc_lsf   = CWrapperNameSpace( "driver" )
cfunc_local = CWrapperNameSpace( "driver" )
cfunc_rsh   = CWrapperNameSpace( "driver" )

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


cfunc_rsh.alloc_driver     = cwrapper.prototype("long    rsh_driver_alloc( char* , char* )")
cfunc_rsh.free_driver      = cwrapper.prototype("void    rsh_driver_free( rsh_driver )")
cfunc_rsh.submit           = cwrapper.prototype("long    rsh_driver_submit_job( rsh_driver , char* , char* , char* , int , char**)")
cfunc_rsh.free_job         = cwrapper.prototype("void    rsh_driver_free_job( job )")
cfunc_rsh.kill_job         = cwrapper.prototype("void    rsh_driver_kill_job( rsh_driver , job )")
cfunc_rsh.add_host         = cwrapper.prototype("void    rsh_driver_add_host( rsh_driver , char* , int)")
cfunc_rsh.get_status       = cwrapper.prototype("int     rsh_driver_get_job_status( rsh_driver , job)")

