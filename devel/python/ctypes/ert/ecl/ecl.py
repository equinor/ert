import time
import datetime
import ctypes
import sys
import os.path
from   ert.cwrap.cwrap       import *
from   ert.job_queue.driver  import LSFDriver , LocalDriver
from   ert.job_queue.driver  import STATUS_PENDING , STATUS_RUNNING , STATUS_DONE , STATUS_EXIT

run_script        = "/project/res/etc/ERT/Scripts/run_eclipse.py"
default_version   = "2009.1"



class Ecl:
    __initialized = False

    sum      = CWrapperNameSpace( "ecl_sum" )
    grid     = CWrapperNameSpace( "ecl_grid" )
    region   = CWrapperNameSpace( "ecl_region" ) 
    ecl_kw   = CWrapperNameSpace( "ecl_kw" )
    ecl_file = CWrapperNameSpace( "ecl_file" )
    ecl_util = CWrapperNameSpace( "ecl_util" )
    
    @classmethod
    def __initialize__(cls):
        if cls.__initialized:
            return

        ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
        cls.libecl = ctypes.CDLL("libecl.so" , ctypes.RTLD_GLOBAL)
        
        cwrapper = CWrapper( cls.libecl )
        cwrapper.registerType( "ecl_sum" , EclSum )
        cls.sum.fread_alloc                   = cwrapper.prototype("long ecl_sum_fread_alloc_case__( char* , char* , bool)") 
        cls.sum.iiget                         = cwrapper.prototype("double ecl_sum_iiget( ecl_sum , int , int)")
        cls.sum.free                          = cwrapper.prototype("void ecl_sum_free( ecl_sum )")
        cls.sum.data_length                   = cwrapper.prototype("int ecl_sum_get_data_length( ecl_sum )")
        cls.sum.iget_sim_days                 = cwrapper.prototype("double ecl_sum_iget_sim_days( ecl_sum , int) ")
        cls.sum.iget_report_step              = cwrapper.prototype("int ecl_sum_iget_report_step( ecl_sum , int) ")
        cls.sum.iget_sim_time                 = cwrapper.prototype("time_t ecl_sum_iget_sim_time( ecl_sum , int) ")
        cls.sum.get_report_end                = cwrapper.prototype("int ecl_sum_iget_report_end( ecl_sum , int)")
        cls.sum.iget_general_var              = cwrapper.prototype("double ecl_sum_get_general_var( ecl_sum , int , char*)")
        cls.sum.get_general_var               = cwrapper.prototype("double ecl_sum_iget_general_var( ecl_sum , int , char*)")
        cls.sum.get_general_var_index         = cwrapper.prototype("int ecl_sum_get_general_var_index( ecl_sum , char*)")
        cls.sum.get_general_var_from_sim_days = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_days( ecl_sum , double , char*)")
        cls.sum.get_general_var_from_sim_time = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_time( ecl_sum , time_t , char*)")

        cls.sum.get_first_gt                  = cwrapper.prototype("int ecl_sum_get_first_gt( ecl_sum , int , double )")
        cls.sum.get_first_lt                  = cwrapper.prototype("int ecl_sum_get_first_lt( ecl_sum , int , double )")
        
        cls.sum.get_start_date                = cwrapper.prototype("time_t ecl_sum_get_start_time( ecl_sum )")
        cls.sum.get_end_date                  = cwrapper.prototype("time_t ecl_sum_get_end_time( ecl_sum )")
        cls.sum.get_last_report_step          = cwrapper.prototype("int ecl_sum_get_last_report_step( ecl_sum )")
        cls.sum.get_first_report_step         = cwrapper.prototype("int ecl_sum_get_first_report_step( ecl_sum )")
        cls.sum.iget_report_step              = cwrapper.prototype("int ecl_sum_iget_report_step( ecl_sum , int )")
        
        ##################################################################

        cwrapper.registerType( "ecl_grid" , EclGrid )
        cls.grid.fread_alloc                  = cwrapper.prototype("long ecl_grid_load_case( char* )")
        cls.grid.free                         = cwrapper.prototype("void ecl_grid_free( ecl_grid )")     
        cls.grid.get_nx                       = cwrapper.prototype("int ecl_grid_get_nx( ecl_grid )")
        cls.grid.get_ny                       = cwrapper.prototype("int ecl_grid_get_ny( ecl_grid )")
        cls.grid.get_nz                       = cwrapper.prototype("int ecl_grid_get_nz( ecl_grid )")
        cls.grid.get_active                   = cwrapper.prototype("int ecl_grid_get_active_size( ecl_grid )")
        cls.grid.get_name                     = cwrapper.prototype("char* ecl_grid_get_name( ecl_grid )") 

        #################################################################

        cwrapper.registerType( "ecl_kw" , EclKW )

        #################################################################

        cwrapper.registerType( "ecl_file" , EclFile )
        cls.ecl_file.fread_alloc               = cwrapper.prototype("long   ecl_file_fread_alloc( char* )")
        cls.ecl_file.free                      = cwrapper.prototype("void   ecl_file_free( ecl_file )")
        cls.ecl_file.iget_kw                   = cwrapper.prototype("ecl_kw ecl_file_iget_kw( ecl_file , int)")
        cls.ecl_file.iget_named_kw             = cwrapper.prototype("ecl_kw ecl_file_iget_named_kw( ecl_file , char* , int)")
                
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

        cls.region.active_size                = cwrapper.prototype("int ecl_region_get_active_size( ecl_region )")
        cls.region.global_size                = cwrapper.prototype("int ecl_region_get_global_size( ecl_region )")

        #################################################################

        cls.ecl_util.get_num_cpu              = cwrapper.prototype("int ecl_util_get_num_cpu( char* )")

        


#################################################################


class EclSumNode:
    def __init__(self , value , report_step , days , time_t):
        self.value       = value  
        self.report_step = report_step
        self.sim_days    = days
        self.sim_time    = time_t


    def __str__(self):
        return "report_step:%d   sim_days:%g    sim_time:%s    value:%g" % (self.report_step , self.sim_days , self.sim_time.date() , self.value)

    

class EclSum:
    def __init__(self , case , join_string = ":" ,include_restart = False):
        self.case            = case
        self.join_string     = join_string
        self.include_restart = include_restart
        self.c_ptr           = 0
        self.reload( )


    def is_valid( self ):
        return not (self.c_ptr == 0)


    def reload(self ):

        """
        Observe that agressive reload() on a running ECLIPSE
        simulation is asking for trouble; the change of finding an
        temporarily incomplete/malformed summary or header file is
        quite large. This will most probably bring the whole thing down.
        """
        c_ptr = Ecl.sum.fread_alloc( self.case , self.join_string , self.include_restart )
        if c_ptr:
            if not self.c_ptr == 0:
                Ecl.sum.free( self )
            self.c_ptr = c_ptr
            
        
    def __del__( self ):
        if self.c_ptr:
            Ecl.sum.free( self )

    def from_param(self):
        return self.c_ptr

    def get_key_index( self , key ):
        return Ecl.sum.get_general_var_index( self , key )

    def __iiget_node(self , key_index , internal_index):
        return EclSumNode( Ecl.sum.iiget( self , internal_index , key_index ) ,
                           Ecl.sum.iget_report_step( self , internal_index )  , 
                           Ecl.sum.iget_sim_days( self , internal_index )     , 
                           Ecl.sum.iget_sim_time( self , internal_index ))


    def __iiget_value(self , key_index , internal_index):
        return Ecl.sum.iiget( self , internal_index , key_index )


    def iget(self , key , internal_index):
        return Ecl.sum.iget_general_var( self , key , internal_index )


    def get_from_days( self , key , days ):
        return Ecl.sum.get_general_var_from_sim_days( self , days , key )

    def get_from_report( self , key , report_step ):
        time_index = Ecl.sum.get_report_end( self , report_step )
        return Ecl.sum.iget_general_var( self , time_index , key )
    
    def get_vector(self , key):
        vector = []
        key_index   = Ecl.sum.get_general_var_index( self , key )
        for internal_index in (range(Ecl.sum.data_length( self ))):
            vector.append( self.__iiget_node( key_index , internal_index ) )
                
        return vector

    
    def iget_days(self , internal_index):
        return Ecl.sum.iget_sim_days( self , internal_index )


    def iget_time(self , internal_index):
        return Ecl.sum.iget_sim_time( self , internal_index )

    def iget_report( self , internal_index ):
        return Ecl.sum.iget_report_step( self , internal_index )

    @property
    def length(self):
        """
        Returns the length of the dataset in terms of many ministeps it contains.
        """
        return Ecl.sum.data_length( self )


    @property
    def start_date(self):
        return Ecl.sum.get_start_date( self )

    @property
    def end_date(self):
        return Ecl.sum.get_end_date( self )

    @property
    def last_report(self):
        return Ecl.sum.get_last_report_step( self )

    @property
    def first_report(self):
        return Ecl.sum.get_first_report_step( self )

    def first_lt(self , key , limit ):
        key_index = Ecl.sum.get_general_var_index( self , key )
        return Ecl.sum.get_first_lt( self , key_index , limit )

    def first_gt(self , key , limit ):
        key_index = Ecl.sum.get_general_var_index( self , key )
        return Ecl.sum.get_first_gt( self , key_index , limit )


#################################################################
class EclKW:
    def __init__(self , c_ptr):
        self.c_ptr = c_ptr

    def from_param(self):
        return self.c_ptr


class EclFile:
    def __init__(self , filename):
        self.c_ptr = Ecl.ecl_file.fread_alloc( filename )

    def __del__(self):
        Ecl.ecl_file.free( self )

    def iget_kw( self , index ):
        return Ecl.ecl_file.iget_kw( self , index )

    def iget_named_kw( self , kw_name , index ):
        return Ecl.ecl_file.iget_named_kw( self , kw_name , index )
        
    def from_param(self):
        return self.c_ptr



class EclGrid:
    def __init__(self , filename):
        self.c_ptr = Ecl.grid.fread_alloc( filename )

    def __del__(self):
        Ecl.grid.free( self )

    def from_param(self):
        return self.c_ptr

    @property
    def nx( self ):
        return Ecl.grid.get_nx( self )

    @property
    def ny( self ):
        return Ecl.grid.get_ny( self )

    @property
    def nz( self ):
        return Ecl.grid.get_nz( self )

    @property
    def active( self ):
        return Ecl.grid.get_active( self )

    @property
    def dims( self ):
        return ( Ecl.grid.get_nx( self ) ,
                 Ecl.grid.get_ny( self ) ,
                 Ecl.grid.get_nz( self ) ,
                 Ecl.grid.get_active( self ) )

    @property
    def name( self ):
        return Ecl.grid.get_name( self )
                 


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

    def active_size( self ):
        return Ecl.region.active_size( self )

    def global_size( self ):
        return Ecl.region.global_size( self )
    
    

#################################################################
class EclCase:
    def __init__(self , input_case):
        self.case = input_case
        (path , tmp) = os.path.split( input_case )
        if path:
            self.path = os.path.abspath( path )
        else:
            self.path = os.getwcwd()
        (self.base , self.ext) = os.path.splitext( tmp )


        self.LSFDriver     = None
        self.LocalDriver   = None
        self.__sum         = None
        self.__grid        = None
        self.__data_file   = None
    

    @property
    def datafile( self ):
        if not self.__data_file:
            if self.path:
                self.__data_file = "%s/%s.DATA" % ( self.path , self.base )
            else:
                self.__data_file = "%s.DATA" % self.base
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

        
    def run( self , version = default_version , blocking = False , run_script = run_script , use_LSF = True ):
        num_cpu = Ecl.ecl_util.get_num_cpu( self.datafile )
        if use_LSF:
            if not self.LSFDriver:
                self.LSFDriver = LSFDriver( )

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



        
    
