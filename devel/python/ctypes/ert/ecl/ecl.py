import time
import datetime
import ctypes
import sys
import os.path


from   ert.cwrap.cwrap       import *
from   ecl_kw                import EclKW , ECL_CHAR_TYPE , ECL_REAL_TYPE , ECL_DOUBLE_TYPE , ECL_INT_TYPE , ECL_BOOL_TYPE , ECL_MESS_TYPE
from   ecl_file              import EclFile
from   fortio                import FortIO
from   ecl_sum               import EclSum
from   ecl_rft               import EclRFTFile , EclRFT , EclRFTCell
from   ecl_grid              import EclGrid
from   ecl_region            import EclRegion
import ecl_util
from   ert.util.tvector      import DoubleVector   
from   ert.util.stringlist   import StringList
#from   ert.util.cfile        import CFILE

from   ert.job_queue.driver  import LSFDriver , LocalDriver
from   ert.job_queue.driver  import STATUS_PENDING , STATUS_RUNNING , STATUS_DONE , STATUS_EXIT


run_script        = "/project/res/etc/ERT/Scripts/run_eclipse.py"
default_version   = "2009.1"



class EclCase:
    def __init__(self , input_case):
        self.case = input_case
        (path , tmp) = os.path.split( input_case )
        if path:
            self.__path = os.path.abspath( path )
        else:
            self.__path = os.getcwd()
        (self.__base , self.ext) = os.path.splitext( tmp )


        self.LSFDriver     = None
        self.LocalDriver   = None
        self.__sum         = None
        self.__grid        = None
        self.__data_file   = None
        self.__rft         = None
    

    @property
    def datafile( self ):
        if not self.__data_file:
            if self.path:
                self.__data_file = "%s/%s.DATA" % ( self.__path , self.__base )
            else:
                self.__data_file = "%s.DATA" % self.__base
        return self.__data_file


    @property
    def sum( self ):
        if not self.__sum:
            self.__sum = EclSum( self.case )
        return self.__sum
    

    @property
    def grid( self ):
        if not self.__grid:
            self.__grid = EclGrid( self.case )
        return self.__grid


    @property
    def rft_file( self ):
        if not self.__rft:
            self.__rft = EclRFTFile( self.case )
        return self.__rft


    @property
    def base( self ):
        return self.__base


    @property
    def path( self ):
        return self.__path
    

        
    def run( self , version = default_version , blocking = False , run_script = run_script , use_LSF = True , LSF_server = None , LSF_queue = "normal"):
        num_cpu = ecl_util.get_num_cpu( self.datafile )
        if use_LSF:
            if not self.LSFDriver:
                self.LSFDriver = LSFDriver( queue      = LSF_queue , 
                                            lsf_server = LSF_server )

            submit_func = self.LSFDriver.submit
            
        else:
            if not self.LocalDriver:
                self.LocalDriver = LocalDriver( )
                
            submit_func = self.LocalDriver.submit
            

        job = submit_func( self.base,
                           run_script , 
                           self.path ,
                           [version , "%s/%s" % (self.path , self.base) , num_cpu ] ,
                           blocking = blocking )

        return job

        




        
    
