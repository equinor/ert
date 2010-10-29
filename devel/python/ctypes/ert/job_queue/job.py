import time
import datetime

# Enum values nicked from libjob_queue/src/basic_queue_driver.h
STATUS_PENDING =  16
STATUS_RUNNING =  32
STATUS_DONE    =  64
STATUS_EXIT    = 128 


class Job:
    def __init__(self , driver , c_ptr , blocking = False):
        self.driver      = driver
        self.c_ptr       = c_ptr
        self.submit_time = datetime.datetime.now()

                    
    def __del__(self):
        self.driver.free_job( self )

    def block( self ):
        while True:
            status = self.status()
            if status == STATUS_DONE or status == STATUS_EXIT:
                break
            else:
                time.sleep( 1 )
    
    def kill( self ):
        self.driver.kill_job( self )
    
    def from_param(self):
        return self.c_ptr

    @property
    def run_time( self ):
        td = datetime.datetime.now() - self.submit_time
        return td.seconds + td.days * 24 * 3600

    @property
    def status( self ):
        return self.driver.get_status( self )

    @property
    def running( self ):
        status = self.driver.get_status( self )
        if status == STATUS_RUNNING:
            return True
        else:
            return False


    @property
    def pending( self ):
        status = self.driver.get_status( self )
        if status == STATUS_PENDING:
            return True
        else:
            return False

    @property
    def complete( self ):
        status = self.driver.get_status( self )
        if status == STATUS_DONE or status == STATUS_EXIT:
            return True
        else:
            return False
        
