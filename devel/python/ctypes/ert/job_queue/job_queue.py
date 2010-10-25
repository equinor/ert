from    ert.cwrap.cwrap       import *
import time
import datetime
import ctypes
import re
import sys
import libjob_queue
import threading
from   driver import *
from   job    import Job
import ert.ecl.ecl_util as ecl_util  




class QueueThread( threading.Thread ):

    def __init__(self , queue):
        threading.Thread.__init__( self )
        self.queue = queue

    def run(self):
        verbose = False
        cfunc.start_queue( self.queue , self.queue.size , verbose )   
        


class JobList:
    def __init__(self):
        self.job_list = []
        self.job_dict = {}
        

    def __getitem__(self , index):
        job = None
        if isinstance(index , types.StringType):
            job = self.job_dict.get( index )
        elif isinstance(index , types.IntType):
            try:
                job = self.job_list[index]
            except:
                job = None
        return job


    def add_job( self , job , job_name ):
        job_index  = len( self.job_list )
        job.job_nr = job_index
        self.job_dict[ job_name ] = job
        self.job_list.append( job )
        

    @property
    def size(self):
        return len( self.job_list )



class exList:
    def __init__(self , joblist):
        self.joblist = joblist

    def __getitem__(self , index):
        job = self.joblist.__getitem__(index)
        if job:
            return True
        else:
            return False

        

class statusList:
    def __init__(self , joblist ):
        self.joblist = joblist

    def __getitem__(self , index):
        job = self.joblist.__getitem__(index)
        if job:
            return job.status()
        else:
            return None


class runtimeList:
    def __init__(self , joblist , queue):
        self.joblist = joblist
        self.queue   = queue

    def __getitem__(self , index):
        job = self.joblist.__getitem__(index)
        if job:
            sim_start = cfunc.iget_sim_start( self.queue , job.job_nr )
            if not sim_start.ctime() == -1:
                return time.time( ) - sim_start.ctime()
            else:
                return None
        else:
            return None


class JobQueue:
    
    def __init__(self , driver , size , max_running , cmd , max_submit = 1):
        OK_file     = None 
        exit_file   = None
        self.c_ptr  = cfunc.alloc_queue( size , max_running , max_submit , False , OK_file , exit_file , cmd )
        self.size   = size
        self.driver = driver
        self.jobs   = JobList()

        self.exists   = exList( self.jobs )
        self.status   = statusList( self.jobs )
        self.run_time = runtimeList( self.jobs , self )
        cfunc.set_driver( self , driver.c_ptr )
        
    
    def kill_job(self , index):
        job = self.jobs.__getitem__( index )
        if job:
            job.kill()

    def start( self , blocking = False):
        self.queue_thread = QueueThread( self )
        self.queue_thread.start()
    

    def __del__(self):
        cfunc.free_queue( self )


    def from_param( self ):
        return self.c_ptr

    def add_job( self , run_path , job_name , argv):
        c_argv = (ctypes.c_char_p * len(argv))()
        c_argv[:] = argv
        job_index = self.jobs.size
        cfunc.insert_job( self , run_path , job_name , job_index , len(argv) , c_argv)
        job = Job( self.driver , cfunc.get_job_ptr( self , job_index ) , False )
        
        self.jobs.add_job( job , job_name )


    def clear( self ):
        pass


    @property
    def running(self):
        return cfunc.is_running( self )

    def num_running( self ):
        return cfunc.num_running( self )

    def num_pending( self ):
        return cfunc.num_pending( self )

    def num_waiting( self ):
        return cfunc.num_waiting( self )

    def num_complete( self ):
        return cfunc.num_complete( self )

    def exists(self , index):
        job = self.__getitem__(index)
        if job:
            return True
        else:
            return False



class EclQueue( JobQueue ):
    default_eclipse_cmd  = "/project/res/etc/ERT/Scripts/run_eclipse.py"
    default_version      = "2009.2"


    def __init__(self , driver , size , max_running , ecl_version = default_version , eclipse_cmd = default_eclipse_cmd):
        JobQueue.__init__( self , driver , size , max_running , eclipse_cmd )
        self.ecl_version = ecl_version


    
    def add_job( self , data_file):
        (path_base , ext) = os.path.splitext( data_file )
        (run_path , base) = os.path.split( path_base )
        
        argv = [ self.ecl_version , path_base , "%s" % ecl_util.get_num_cpu( data_file )]
        JobQueue.add_job( self , run_path , base , argv)


#################################################################

cwrapper = CWrapper( libjob_queue.lib )
cwrapper.registerType( "job_queue" , JobQueue )
cfunc  = CWrapperNameSpace( "JobQeueu" )

cfunc.alloc_queue     = cwrapper.prototype("long job_queue_alloc(int , int , int , bool , char* )")
cfunc.free_queue      = cwrapper.prototype("void job_queue_free( job_queue )")
cfunc.set_max_running = cwrapper.prototype("void job_queue_set_max_running( job_queue , int)")
cfunc.get_max_running = cwrapper.prototype("int  job_queue_get_max_running( job_queue )")
cfunc.set_driver      = cwrapper.prototype("void job_queue_set_driver( job_queue , long )")
cfunc.insert_job      = cwrapper.prototype("void job_queue_insert_job( job_queue , char* , char* , int , long)")
cfunc.start_queue     = cwrapper.prototype("void job_queue_run_jobs( job_queue , int , bool)")
cfunc.num_running     = cwrapper.prototype("int  job_queue_get_num_running( job_queue )")
cfunc.num_complete    = cwrapper.prototype("int  job_queue_get_num_complete( job_queue )")
cfunc.num_waiting     = cwrapper.prototype("int  job_queue_get_num_waiting( job_queue )")
cfunc.num_pending     = cwrapper.prototype("int  job_queue_get_num_pending( job_queue )")
cfunc.is_running      = cwrapper.prototype("int  job_queue_is_running( job_queue )")
cfunc.get_job_ptr     = cwrapper.prototype("long job_queue_iget_job_data( job_queue , int)")
cfunc.iget_sim_start  = cwrapper.prototype("time_t job_queue_iget_sim_start( job_queue , int)")
