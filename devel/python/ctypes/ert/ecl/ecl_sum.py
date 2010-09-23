import ctypes
import numpy
import matplotlib.dates
import libecl
from   ert.cwrap.cwrap       import *
from   ert.util.stringlist   import StringList


class EclSumNode:
    def __init__(self , value , report_step , days , time_t):
        self.value       = value  
        self.report_step = report_step
        self.sim_days    = days
        self.sim_time    = time_t


    def __str__(self):
        return "report_step:%d   sim_days:%g    sim_time:%s    value:%g" % (self.report_step , self.sim_days , self.sim_time.date() , self.value)

    


class EclSum(object):
    
    def __new__( cls , case , join_string = ":" , include_restart = True):
        """
        The constructor loads a summary case, if no summary can be loaded the constructor will
        return None.
        """
        c_ptr = cfunc.fread_alloc( case , join_string , include_restart)
        if c_ptr:
            obj = object.__new__( cls )
            obj.c_ptr = c_ptr
            return obj
        else:
            return None


    def __init__(self , case , join_string = ":" ,include_restart = False , c_ptr = None):
        self.case            = case
        self.join_string     = join_string
        self.include_restart = include_restart


    def __del__( self ):
        if self.c_ptr:
            cfunc.free( self )

    def from_param(self):
        return self.c_ptr

    def get_key_index( self , key ):
        index = cfunc.get_general_var_index( self , key )
        if index >= 0:
            return index
        else:
            return None


    def __iiget_node(self , key_index , internal_index):
        return EclSumNode( cfunc.iiget( self , internal_index , key_index ) ,
                           cfunc.iget_report_step( self , internal_index )  , 
                           cfunc.iget_sim_days( self , internal_index )     , 
                           cfunc.iget_sim_time( self , internal_index ))


    def __iiget_value(self , key_index , internal_index):
        return cfunc.iiget( self , internal_index , key_index )


    def iget(self , key , internal_index):
        return cfunc.iget_general_var( self , key , internal_index )

    def get_interp( self , key , days = None , time = None):
        if days:
            return cfunc.get_general_var_from_sim_days( self , days , key )
        else:
            return cfunc.get_general_var_from_sim_time( self , time , key )

    def get_from_report( self , key , report_step ):
        time_index = cfunc.get_report_end( self , report_step )
        return cfunc.iget_general_var( self , time_index , key )
    
    def get_vector(self , key):
        key_index   = cfunc.get_general_var_index( self , key )
        if key_index:
            vector = []
            for internal_index in (range(cfunc.data_length( self ))):
                vector.append( self.__iiget_node( key_index , internal_index ) )
            return vector
        else:
            return None


    def numpy_value( self , key ):
        key_index = cfunc.get_general_var_index( self , key )
        if key_index:
            length = cfunc.data_length( self )
            value = numpy.zeros( length )
            for i in range(length ):
                value[i] = cfunc.iiget( self , i , key_index )
            return value
        else:
            return None

    @property
    def numpy_days( self ):
        length = cfunc.data_length( self )
        days = numpy.zeros( length )
        for i in range( length ):
            days[i] = cfunc.iget_sim_days( self , i)
        return days

    @property
    def numpy_mpl_time( self ):
        length = cfunc.data_length( self )
        time   = numpy.zeros( length )
        for i in range( length ):
            ctime = cfunc.iget_sim_time( self , i)
            time[i] = matplotlib.dates.date2num( ctime.datetime() )
        return time


    def iget_days(self , internal_index):
        return cfunc.iget_sim_days( self , internal_index )


    def iget_time(self , internal_index):
        return cfunc.iget_sim_time( self , internal_index )

    def iget_report( self , internal_index ):
        return cfunc.iget_report_step( self , internal_index )

    @property
    def length(self):
        """
        Returns the length of the dataset in terms of many ministeps it contains.
        """
        return cfunc.data_length( self )


    def select_matching( self , pattern ):
        s = StringList()
        cfunc.select_matching_keys( self , pattern , s )
        return s.strings


    @property
    def start_date(self):
        return cfunc.get_start_date( self )

    @property
    def end_date(self):
        return cfunc.get_end_date( self )

    @property
    def last_report(self):
        return cfunc.get_last_report_step( self )

    @property
    def first_report(self):
        return cfunc.get_first_report_step( self )

    def first_lt(self , key , limit ):
        key_index = cfunc.get_general_var_index( self , key )
        return cfunc.get_first_lt( self , key_index , limit )

    def first_gt(self , key , limit ):
        key_index = cfunc.get_general_var_index( self , key )
        return cfunc.get_first_gt( self , key_index , limit )




#################################################################

# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_sum" , EclSum )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_sum")


cfunc.fread_alloc                   = cwrapper.prototype("long ecl_sum_fread_alloc_case__( char* , char* , bool)") 
cfunc.iiget                         = cwrapper.prototype("double ecl_sum_iiget( ecl_sum , int , int)")
cfunc.free                          = cwrapper.prototype("void ecl_sum_free( ecl_sum )")
cfunc.data_length                   = cwrapper.prototype("int ecl_sum_get_data_length( ecl_sum )")
cfunc.iget_sim_days                 = cwrapper.prototype("double ecl_sum_iget_sim_days( ecl_sum , int) ")
cfunc.iget_report_step              = cwrapper.prototype("int ecl_sum_iget_report_step( ecl_sum , int) ")
cfunc.iget_sim_time                 = cwrapper.prototype("time_t ecl_sum_iget_sim_time( ecl_sum , int) ")
cfunc.get_report_end                = cwrapper.prototype("int ecl_sum_iget_report_end( ecl_sum , int)")
cfunc.iget_general_var              = cwrapper.prototype("double ecl_sum_get_general_var( ecl_sum , int , char*)")
cfunc.get_general_var               = cwrapper.prototype("double ecl_sum_iget_general_var( ecl_sum , int , char*)")
cfunc.get_general_var_index         = cwrapper.prototype("int ecl_sum_get_general_var_index( ecl_sum , char*)")
cfunc.get_general_var_from_sim_days = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_days( ecl_sum , double , char*)")
cfunc.get_general_var_from_sim_time = cwrapper.prototype("double ecl_sum_get_general_var_from_sim_time( ecl_sum , time_t , char*)")
cfunc.get_first_gt                  = cwrapper.prototype("int ecl_sum_get_first_gt( ecl_sum , int , double )")
cfunc.get_first_lt                  = cwrapper.prototype("int ecl_sum_get_first_lt( ecl_sum , int , double )")
cfunc.get_start_date                = cwrapper.prototype("time_t ecl_sum_get_start_time( ecl_sum )")
cfunc.get_end_date                  = cwrapper.prototype("time_t ecl_sum_get_end_time( ecl_sum )")
cfunc.get_last_report_step          = cwrapper.prototype("int ecl_sum_get_last_report_step( ecl_sum )")
cfunc.get_first_report_step         = cwrapper.prototype("int ecl_sum_get_first_report_step( ecl_sum )")
cfunc.iget_report_step              = cwrapper.prototype("int  ecl_sum_iget_report_step( ecl_sum , int )")
cfunc.select_matching_keys          = cwrapper.prototype("void ecl_sum_select_matching_general_var_list( ecl_sum , char* , stringlist )")
