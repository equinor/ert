import time
import datetime
import ctypes
import sys
import os.path

from   ert.cwrap.cwrap       import *
import ert.util.ctime        
import ecl_util
import ecl_default
import ecl_grav
from   ecl_kw                import ECL_INT_TYPE , ECL_FLOAT_TYPE , ECL_CHAR_TYPE , ECL_BOOL_TYPE , ECL_DOUBLE_TYPE, EclKW
from   ecl_file              import EclFile
from   ecl_sum               import EclSum
from   ecl_rft               import EclRFTFile , EclRFT , EclRFTCell
from   ecl_grid              import EclGrid
from   ecl_region            import EclRegion
from   ecl_util              import *  


class EclCase:
    def __init__(self , input_case):
        self.case = input_case
        (path , tmp) = os.path.split( input_case )
        if path:
            self.__path = os.path.abspath( path )
        else:
            self.__path = os.getcwd()
        (self.__base , self.ext) = os.path.splitext( tmp )
        
        
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

        
    def run( self , driver , ecl_version = ecl_default.version , blocking = False ):
        job = driver.submit_ecl( self.datafile , blocking = blocking)
        return job
        

    def submit( self , queue ):
        queue.add_job( self.datafile )
        



        
    
