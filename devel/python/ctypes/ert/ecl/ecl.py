import time
import datetime
import ctypes
import sys
import os.path


from   ert.cwrap.cwrap       import *

from   ecl_kw                import EclKW
from   ecl_file              import EclFile
from   fortio                import FortIO
from   ecl_sum               import EclSum
from   ecl_grid              import EclGrid
from   ert.util.tvector      import DoubleVector   
from   ert.util.stringlist   import StringList


from   ert.job_queue.driver  import LSFDriver , LocalDriver
from   ert.job_queue.driver  import STATUS_PENDING , STATUS_RUNNING , STATUS_DONE , STATUS_EXIT


RFT = 1
PLT = 2

run_script        = "/project/res/etc/ERT/Scripts/run_eclipse.py"
default_version   = "2009.1"




class Ecl:
    __initialized = False

    sum      = CWrapperNameSpace( "ecl_sum" )
    grid     = CWrapperNameSpace( "ecl_grid" )
    region   = CWrapperNameSpace( "ecl_region" ) 
    ecl_util = CWrapperNameSpace( "ecl_util" )
    rft_file = CWrapperNameSpace(" ecl_rft_file ")
    rft      = CWrapperNameSpace(" ecl_rft_node ")
    
    @classmethod
    def __initialize__(cls):
        if cls.__initialized:
            return

        ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
        cls.libutil = ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
        cls.libecl =  ctypes.CDLL("libecl.so" , ctypes.RTLD_GLOBAL)
        
        cwrapper = CWrapper( cls.libecl )

        #################################################################


        cwrapper.registerType( "ecl_region" , EclRegion )
        cls.region.alloc                      = cwrapper.prototype("long ecl_region_alloc( ecl_grid , bool )")
        cls.region.free                       = cwrapper.prototype("void ecl_region_free( ecl_region )")     
        cls.region.reset                      = cwrapper.prototype("void ecl_region_reset( ecl_region )")

        cls.region.select_all                 = cwrapper.prototype("void ecl_region_select_all( ecl_region )")
        cls.region.deselect_all               = cwrapper.prototype("void ecl_region_deselect_all( ecl_region )")

        cls.region.select_equal               = cwrapper.prototype("void ecl_region_select_all( ecl_region , ecl_kw , int )")
        cls.region.deselect_equal             = cwrapper.prototype("void ecl_region_deselect_all( ecl_region , ecl_kw , int)")

        cls.region.select_less                = cwrapper.prototype("void ecl_region_select_smaller( ecl_region , ecl_kw , float )")
        cls.region.deselect_less              = cwrapper.prototype("void ecl_region_deselect_smaller( ecl_region , ecl_kw , float )")

        cls.region.select_more                = cwrapper.prototype("void ecl_region_select_larger( ecl_region , ecl_kw , float )")
        cls.region.deselect_more              = cwrapper.prototype("void ecl_region_deselect_larger( ecl_region , ecl_kw , float )")

        cls.region.invert_selection           = cwrapper.prototype("void ecl_region_invert_selection( ecl_region )")

        cls.region.active_size                = cwrapper.prototype("int  ecl_region_get_active_size( ecl_region )")
        cls.region.global_size                = cwrapper.prototype("int  ecl_region_get_global_size( ecl_region )")
        cls.region.active_set                 = cwrapper.prototype("int* ecl_region_get_active_list( ecl_region )")
        cls.region.global_set                 = cwrapper.prototype("int* ecl_region_get_global_list( ecl_region )")

        #################################################################

        cls.ecl_util.get_num_cpu              = cwrapper.prototype("int ecl_util_get_num_cpu( char* )")

        #################################################################
        cwrapper.registerType( "ecl_rft_file" , EclRFTFile )
        cwrapper.registerType( "ecl_rft"      , EclRFT )
        
        cls.rft_file.load                     = cwrapper.prototype("long ecl_rft_file_alloc_case( char* )")
        cls.rft_file.has_rft                  = cwrapper.prototype("bool ecl_rft_file_case_has_rft( char* )")
        cls.rft_file.free                     = cwrapper.prototype("void ecl_rft_file_free( ecl_rft_file )")
        cls.rft_file.get_size                 = cwrapper.prototype("int ecl_rft_file_get_size__( ecl_rft_file , char* , time_t)")
        cls.rft_file.iget                     = cwrapper.prototype("long ecl_rft_file_iget_node( ecl_rft_file , int )")
        cls.rft_file.get_num_wells            = cwrapper.prototype("int  ecl_rft_file_get_num_wells( ecl_rft_file )")
        cls.rft_file.get_rft                  = cwrapper.prototype("long ecl_rft_file_get_well_time_rft( ecl_rft_file , char* , time_t)")

        cls.rft.get_type                      = cwrapper.prototype("int    ecl_rft_node_get_type( ecl_rft )")
        cls.rft.get_well                      = cwrapper.prototype("char*  ecl_rft_node_get_well_name( ecl_rft )")
        cls.rft.get_date                      = cwrapper.prototype("time_t ecl_rft_node_get_date( ecl_rft )")
        cls.rft.get_size                      = cwrapper.prototype("int ecl_rft_node_get_size( ecl_rft )")
        cls.rft.iget_depth                    = cwrapper.prototype("double ecl_rft_node_iget_depth( ecl_rft )")
        cls.rft.iget_pressure                 = cwrapper.prototype("double ecl_rft_node_iget_pressure(ecl_rft)")
        cls.rft.iget_ijk                      = cwrapper.prototype("void ecl_rft_node_iget_ijk( ecl_rft , int , int*, int*, int*)") 
        cls.rft.iget_swat                     = cwrapper.prototype("double ecl_rft_node_iget_swat(ecl_rft)")
        cls.rft.iget_sgas                     = cwrapper.prototype("double ecl_rft_node_iget_sgas(ecl_rft)")
        cls.rft.iget_orat                     = cwrapper.prototype("double ecl_rft_node_iget_orat(ecl_rft)")
        cls.rft.iget_wrat                     = cwrapper.prototype("double ecl_rft_node_iget_wrat(ecl_rft)")
        cls.rft.iget_grat                     = cwrapper.prototype("double ecl_rft_node_iget_grat(ecl_rft)")
        cls.rft.lookup_ijk                    = cwrapper.prototype("int    ecl_rft_node_lookup_ijk( ecl_rft , int , int , int)")
        
        #################################################################
        
        

class EclRFTFile(object):
    def __new__( cls , case ):
        c_ptr = Ecl.rft_file.load( case )
        if c_ptr:
            obj = object.__new__( cls )
            obj.c_ptr = c_ptr
            return obj
        else:
            return None

    def __del__(self):
        Ecl.rft_file.free( self )

    def from_param( self ):
        return self.c_ptr
    
    @property
    def size( self , well = None , date = None):
        return Ecl.rft_file.get_size( self , well , -1)

    @property
    def num_wells( self ):
        return Ecl.rft_file.get_num_wells( self )

    @property
    def headers(self):
        header_list = []
        for i in (range(Ecl.rft_file.get_size( self , None , -1))):
            rft = self.iget( i )
            print rft
            header_list.append( (rft.well , rft.date) )
        return header_list

    def iget(self , index):
        return EclRFT( Ecl.rft_file.iget( self , index ) , self )

    def get(self , well_name , date ):
        c_ptr = Ecl.rft_file.get_rft( self , well_name , ctime( date )) 
        if c_ptr:
            return EclRFT( c_ptr , self)
        else:
            return None


class EclRFT:
    def __init__(self , c_ptr , parent):
        self.c_ptr  = c_ptr
        self.parent = parent    # Inhibit GC

    def from_param( self ):
        return self.c_ptr

    @property
    def type(self):
        # Enum: ecl_rft_enum from ecl_rft_node.h
        # RFT     = 1
        # PLT     = 2
        # Segment = 3  -- Not properly implemented
        return Ecl.rft.get_type( self )

    @property
    def well(self):
        return Ecl.rft.get_well( self )

    @property
    def date(self):
        return Ecl.rft.get_date( self )

    @property
    def size(self):
        return Ecl.rft.get_size( self )


    def iget( self, index):
        i = ctypes.c_int()
        j = ctypes.c_int()
        k = ctypes.c_int()
        pressure = Ecl.rft.iget_pressure( self, index )
        depth    = Ecl.rft.iget_depth( self, index )
        Ecl.rft.iget_ijk( self, index , ctypes.byref(i), ctypes.byref(j) , ctypes.byref(k))

        
        if self.type == RFT:
            swat = Ecl.rft.iget_swat( self, index )
            sgas = Ecl.rft.iget_sgas( self, index )
            return EclRFTCell.RFTCell( i,j,k,depth , pressure,swat,sgas)
        else:
            orat = Ecl.rft.iget_orat( self, index )
            wrat = Ecl.rft.iget_wrat( self, index )
            grat = Ecl.rft.iget_grat( self, index )
            return EclRFTCell.PLTCell( i,j,k,depth , pressure,orat,grat,wrat)

    # ijk are zero offset
    def ijkget( self , ijk ):
        index = Ecl.rft.lookup_ijk( self , ijk[0] , ijk[1] , ijk[2])
        if index >= 0:
            return self.iget( index )
        else:
            return None



class EclRFTCell:
    def __init__(self , type , i , j , k , depth , pressure):
        self.type = type
        self.__i = i
        self.__j = j
        self.__k = k
        self.__depth    = depth
        self.__pressure = pressure

        self.__sgas     = None 
        self.__swat     = None
        self.__orat     = None
        self.__wrat     = None
        self.__grat     = None

    @classmethod
    def RFTCell( cls , i,j,k,depth,pressure,swat,sgas):
        cell = cls(RFT , i,j,k,depth,pressure)
        cell.__swat = swat
        cell.__sgas = sgas
        return cell

    @classmethod
    def PLTCell(cls , i,j,k,depth,pressure,orat,grat,wrat):
        cell = cls(PLT , i,j,k,depth,pressure)
        cell.__orat = orat
        cell.__wrat = wrat
        cell.__grat = grat
        return cell
        
    @property
    def i(self):
        return self.__i.value

    @property
    def j(self):
        return self.__j.value

    @property
    def k(self):
        return self.__k.value

    @property
    def ijk(self):
        return (self.__i.value , self.__j.value , self.__k.value)

    @property
    def pressure(self):
        return self.__pressure
    
    @property
    def depth(self):
        return self.__depth

    @property
    def PLT(self):
        if self.type == PLT:
            return True
        else:
            return False

    @property
    def sgas(self):
        return self.__sgas

    @property
    def swat(self):
        return self.__swat

    @property
    def orat(self):
        return self.__orat

    @property
    def grat(self):
        return self.__grat

    @property
    def wrat(self):
        return self.__wrat




    
class EclRegion:
    def __init__(self , grid , preselect):
        self.grid  = grid
        self.c_ptr = Ecl.region.alloc( grid , preselect )
        
    def __del__( self ):
        Ecl.region.free( self )

    def from_param(self):
        return self.c_ptr

    def reset(self):
        Ecl.region.reset( self )

    def select_more( self , ecl_kw , limit):
        Ecl.region.select_more( self , ecl_kw , limit )

    def select_less( self , ecl_kw , limit):
        Ecl.region.select_less( self , ecl_kw , limit )

    def select_equal( self , ecl_kw , value ):
        Ecl.region.select_equal( self , ecl_kw , value )

    def select_all( self ):
        Ecl.region.select_all( self )

    def invert( self ):
        Ecl.region.invert_selection( self )

    def active_size( self ):
        return Ecl.region.active_size( self )

    def global_size( self ):
        return Ecl.region.global_size( self )
    
    
    def active_set( self ):
        list = Ecl.region.active_set( self )
        list.size = Ecl.region.active_size( self )
        return list

    
    def global_set( self ):
        list = Ecl.region.global_set( self )
        list.size = Ecl.region.global_size( self )
        return list
        
    
    

#################################################################
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
            if Ecl.rft_file.has_rft( self.case ):
                self.__rft = EclRFTFile( self.case )
            else:
                print "Does not have RFT??"
                
        return self.__rft


    @property
    def base( self ):
        return self.__base


    @property
    def path( self ):
        return self.__path
    

        
    def run( self , version = default_version , blocking = False , run_script = run_script , use_LSF = True , LSF_server = None , LSF_queue = "normal"):
        num_cpu = Ecl.ecl_util.get_num_cpu( self.datafile )
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

        




Ecl.__initialize__() # Run once



        
    
