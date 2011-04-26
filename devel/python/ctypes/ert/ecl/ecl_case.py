#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_case.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import os.path

import ecl_sum
import ecl_grid
import ecl_rft
import ecl_default

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
            self.__sum = ecl_sum.EclSum( self.case )
        return self.__sum
    

    @property
    def grid( self ):
        if not self.__grid:
            self.__grid = ecl_grid.EclGrid( self.case )
        return self.__grid


    @property
    def rft_file( self ):
        if not self.__rft:
            self.__rft = ecl_rft.EclRFTFile( self.case )
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
