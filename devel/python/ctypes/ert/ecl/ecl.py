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
        
        cls.sum.test1                         = cwrapper.prototype("void test1(int , char**)")
        cls.sum.test2                         = cwrapper.prototype("void test2(char**)")

        ##################################################################

        cwrapper.registerType( "ecl_grid" , EclGrid )
        cls.grid.fread_alloc                  = cwrapper.prototype("long ecl_grid_alloc( char* )")
        cls.grid.free                         = cwrapper.prototype("void ecl_grid_free( ecl_grid )")     

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


class EclSumNode:
    def __init__(self , value , report_step , days , time_t):
        self.value       = value  
        self.report_step = report_step
        self.sim_days    = days
        self.sim_time    = time_t


    def __str__(self):
        return "report_step:%d   sim_days:%g    sim_time:%s    value:%g" % (self.report_step , self.sim_days , self.sim_time.date() , self.value)

    

def test1( *L ):
    arr = (ctypes.c_char_p * len(L))()
    for i in (range(len(L))):
        arr[i] = L[i]

    Ecl.sum.test1( len(L) , arr )


def test2( L ):
    arr = (ctypes.c_char_p * (len(L) + 1))()
    arr[:-1] = L
    arr[ len(L) ] = None
    Ecl.sum.test2(  arr )




class EclSum:
    def __init__(self , case , join_string = ":" ,include_restart = False):
        self.case      = case
        self.join_string = join_string
        self.include_restart = include_restart
        self.c_ptr = None
        self.reload( )
        

    def reload(self ):
        if self.c_ptr != None:
            Ecl.sum.free( self )
        self.c_ptr = Ecl.sum.fread_alloc( self.case , self.join_string , self.include_restart)
        
        
    def __del__( self ):
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


    def length(self):
        """
        Returns the length of the dataset in terms of many ministeps it contains.
        """
        return Ecl.sum.data_length( self )

    
    def iget_days(self , internal_index):
        return Ecl.sum.iget_sim_days( self , internal_index )


    def iget_time(self , internal_index):
        return Ecl.sum.iget_sim_time( self , internal_index )

    def iget_report( self , internal_index ):
        return Ecl.sum.iget_report_step( self , internal_index )

    def start_date(self):
        return Ecl.sum.get_start_date( self )

    def end_date(self):
        return Ecl.sum.get_end_date( self )

    def last_report(self):
        return Ecl.sum.get_last_report_step( self )

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



class EclRegion:
    def __init__(self , grid , preselect):
        self.grid  = grid
        self.c_ptr = Ecl.region.alloc( grid , preselect )
        
    def __del__( self ):
        Ecl.region.free( self )

    def from_param(self):
        return self.c_ptr

    def select_more( self , ecl_kw , limit):
        Ecl.region.select_more( self , ecl_kw , limit )

    def select_less( self , ecl_kw , limit):
        Ecl.region.select_less( self , ecl_kw , limit )

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
        self.LSFDriver   = None
        self.LocalDriver = None
        
        
        
    def run( self , version = default_version , num_cpu = 1 , blocking = False , run_script = run_script , use_LSF = True ):
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



        
    
