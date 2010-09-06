import time
import datetime
import ctypes
import sys
import os
from   ert.cwrap.cwrap import *

# Enum values nicked from libjob_queue/src/basic_queue_driver.h
STATUS_PENDING =  16
STATUS_RUNNING =  32
STATUS_DONE    =  64
STATUS_EXIT    = 128 

LSF_HOME    = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib"
LAPACK_HOME = "/project/res/x86_64_RH_4/lib"

class DriverContext:
    __initialized = False
    @classmethod
    def __initialize__(cls):
        if cls.__initialized:
            return
        
        ctypes.CDLL("libnsl.so"                     , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("%s/liblsf.so" % LSF_HOME       , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("%s/libbat.so" % LSF_HOME       , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libz.so"                       , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("%s/liblapack.so" % LAPACK_HOME , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libutil.so"                    , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libconfig.so"                  , ctypes.RTLD_GLOBAL)
        cls.__libjob_queue = ctypes.CDLL("libjob_queue.so" , ctypes.RTLD_GLOBAL)

        cls.cwrapper = CWrapper( cls.__libjob_queue )
        cls.cwrapper.registerType( "job"   , Job )
    

    @classmethod
    def get_cwrapper( cls ):
        if not cls.__initialized:
            cls.__initialize__()
        return cls.cwrapper

                


class LSFDriver:

    lsf = CWrapperNameSpace("LSF")

    @classmethod
    def __initialize__( cls ):
        cwrapper = DriverContext.get_cwrapper()
        cwrapper.registerType( "lsf_driver" , LSFDriver )
        
        cls.lsf.alloc_driver = cwrapper.prototype("long    lsf_driver_alloc( char* , char* , char* , int )")
        cls.lsf.free_driver  = cwrapper.prototype("void    lsf_driver_free( lsf_driver )")
        cls.lsf.submit       = cwrapper.prototype("long    lsf_driver_submit_job( lsf_driver , char* , char* , char* , int , char**)")
        cls.lsf.free_job     = cwrapper.prototype("void    lsf_driver_free_job( job )")
        cls.lsf.get_status   = cwrapper.prototype("int     lsf_driver_get_job_status( lsf_driver , job)")
        cls.lsf.kill_job     = cwrapper.prototype("void    lsf_driver_kill_job( lsf_driver , job )")
        

    def from_param(self):
        return self.c_ptr


    def __init__(self ,
                 queue = "normal" ,
                 resource_request = "select[cs && x86_64Linux] rusage[ecl100v2000=1:duration=5]" ,
                 remote_server = None ):
        if not remote_server:
            os.environ["ERT_LINK_LSF"] = "True"
        self.c_ptr = self.lsf.alloc_driver( queue , resource_request , remote_server , 1)


    def __del__( self ):
        self.lsf.free_driver( self )


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = self.lsf.submit( self , cmd , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )

        return job
        

    def free_job( self , job ):
        self.lsf.free_job( job )
    

    def get_status( self , job ):
        return self.lsf.get_status( self , job )
    

    def kill_job( self , job ):
        self.lsf.kill_job( self , job )


        

class LocalDriver:
    local = CWrapperNameSpace("Local")
    @classmethod
    def __initialize__( cls ):
        cwrapper = DriverContext.get_cwrapper()
        cwrapper.registerType( "local_driver" , LocalDriver )

        cls.local.alloc_driver     = cwrapper.prototype("long    local_driver_alloc( )")
        cls.local.free_driver      = cwrapper.prototype("void    local_driver_free( local_driver )")
        cls.local.submit           = cwrapper.prototype("long    local_driver_submit_job( local_driver , char* , char* , char* , int , char**)")
        cls.local.free_job         = cwrapper.prototype("void    local_driver_free_job( job )")
        cls.local.kill_job         = cwrapper.prototype("void    local_driver_kill_job( local_driver , job )")
        cls.local.get_status       = cwrapper.prototype("int     local_driver_get_job_status( local_driver , job)")


    def __init__( self ):
        self.c_ptr = self.local.alloc_driver( )


    def __del__( self ):
        self.local.free_driver( self )

    def from_param(self):
        return self.c_ptr


    def submit( self , name , cmd , run_path , argList , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        c_ptr = self.local.submit( self , cmd , run_path , name , argc , argv )
        job = Job( self , c_ptr , blocking)
        return job
        

    def free_job( self , job ):
        self.local.free_job( job )
    

    def get_status( self , job ):
        return self.local.get_status( self , job )
    

    def kill_job( self , job ):
        self.local.kill_job( self , job )
    
        
        

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



LSFDriver.__initialize__()
LocalDriver.__initialize__()
