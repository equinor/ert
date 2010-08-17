import time
import datetime
import ctypes
import sys
from   ert.cwrap.cwrap import *




class Ecl:
    __initialized = False

    sum = CWrapperNameSpace( "ecl_sum" )
    
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
        cls.sum.fread_alloc                   = cwrapper.prototype("long ecl_sum_fread_alloc_case( char* , char* )") 
        cls.sum.iiget                         = cwrapper.prototype("double ecl_sum_iiget( ecl_sum , int , int)")
        cls.sum.free                          = cwrapper.prototype("void ecl_sum_free( ecl_sum )")
        cls.sum.data_length                   = cwrapper.prototype("int ecl_sum_get_data_length( ecl_sum )")
        cls.sum.iget_sim_days                 = cwrapper.prototype("double ecl_sum_iget_sim_days( ecl_sum , int) ")
        cls.sum.iget_report_step              = cwrapper.prototype("int ecl_sum_iget_report_step( ecl_sum , int) ")
        cls.sum.iget_sim_time                 = cwrapper.prototype("time_t ecl_sum_iget_sim_time( ecl_sum , int) ")
        cls.sum.get_report_ministep_end       = cwrapper.prototype("int ecl_sum_get_report_ministep_end( ecl_sum , int)")
        cls.sum.iget_general_var              = cwrapper.prototype("double ecl_sum_get_general_var( ecl_sum , int , char*)")
        cls.sum.get_general_var               = cwrapper.prototype("double ecl_sum_get_general_var( ecl_sum , int , char*)")
        cls.sum.get_general_var_index         = cwrapper.prototype("int ecl_sum_get_general_var_index( ecl_sum , char*)")
        cls.sum.get_general_var_from_sim_days = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_days( ecl_sum , double , char*)")
        cls.sum.get_general_var_from_sim_time = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_time( ecl_sum , time_t , char*)")

        cls.sum.get_first_gt                  = cwrapper.prototype("int ecl_sum_get_first_gt( ecl_sum , int , double )")
        cls.sum.get_first_lt                  = cwrapper.prototype("int ecl_sum_get_first_lt( ecl_sum , int , double )")
        
        cls.sum.get_start_date                = cwrapper.prototype("time_t ecl_sum_get_start_time( ecl_sum )")
        cls.sum.get_end_date                  = cwrapper.prototype("time_t ecl_sum_get_end_time( ecl_sum )")
        cls.sum.get_last_report_step          = cwrapper.prototype("int ecl_sum_get_last_report_step( ecl_sum )")
        cls.sum.get_first_report_step          = cwrapper.prototype("int ecl_sum_get_first_report_step( ecl_sum )")
        

        cls.sum.test1 = cls.libecl.test1
        cls.sum.test1.restype = None
        cls.sum.test1.argtypes = [ ctypes.c_int , ctypes.POINTER( ctypes.c_char_p ) ]

        cls.sum.test2 = cls.libecl.test2
        cls.sum.test2.restype = None
        cls.sum.test2.argtypes = [ ctypes.POINTER( ctypes.c_char_p ) ]

        



class EclSumNode:
    def __init__(self , value , report_step , days , time_t):
        self.value       = value  
        self.report_step = report_step
        self.sim_days    = days
        self.sim_time    = time_t


    def __str__(self):
        return "report_step:%d   sim_days:%g    sim_time:%s    value:%g" % (self.report_step , self.sim_days , self.sim_time.date() , self.value)

    

#def test1( L ):
#    arr = (ctypes.c_char_p * len(L))()
#    arr[:] = L
#    Ecl.sum.test1( len(L) , arr )
#
#def test2(   L ):
#    arr = (ctypes.c_char_p * (len(L) + 1))()
#    arr[:-1] = L
#    arr[ len(L) ] = None
#    Ecl.sum.test2(  arr )
    

class EclSum:
    def __init__(self , case , join_string = ":"):
        self.c_ptr     = Ecl.sum.fread_alloc( case , join_string )
        self.key_cache = {}
        
    def __del__( self ):
        Ecl.sum.free( self )


    def from_param(self):
        return self.c_ptr

    def get_key_index( self , key ):
        if not self.key_cache.has_key( key ):
            self.key_cache[key] = Ecl.sum.get_general_var_index( self , key )
        return self.key_cache[key]



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
        ministep = Ecl.sum.get_report_ministep_end( self , report_step )
        return Ecl.sum.get_general_var( self , ministep , key )

        
    def get(self , key , ministep = None, report_step = None, days = None, sim_time = None):
        if report_step:
            ministep = Ecl.sum.get_report_ministep_end( self , report_step )

        if ministep:
            return Ecl.sum.get_general_var( self , ministep , key ) 
        else:
            sys.exit("??")


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



Ecl.__initialize__() # Run once
__libecl = Ecl.libecl



        
    
