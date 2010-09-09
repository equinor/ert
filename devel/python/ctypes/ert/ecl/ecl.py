import time
import datetime
import ctypes
import sys
import os.path
from   ert.cwrap.cwrap       import *
from   ert.job_queue.driver  import LSFDriver , LocalDriver
from   ert.job_queue.driver  import STATUS_PENDING , STATUS_RUNNING , STATUS_DONE , STATUS_EXIT


RFT = 1
PLT = 2

# Enum defintion from ecl_util.h
ECL_CHAR_TYPE   = 0
ECL_REAL_TYPE   = 1
ECL_DOUBLE_TYPE = 2
ECL_INT_TYPE    = 3
ECL_BOOL_TYPE   = 4
ECL_MESS_TYPE   = 5



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
    rft_file = CWrapperNameSpace(" ecl_rft_file ")
    rft      = CWrapperNameSpace(" ecl_rft_node ")
    
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
        cwrapper.registerType( "ecl_kw"  , EclKW )
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
        cls.grid.exists                       = cwrapper.prototype("bool ecl_grid_exists( char* )")
        cls.grid.free                         = cwrapper.prototype("void ecl_grid_free( ecl_grid )")     
        cls.grid.get_nx                       = cwrapper.prototype("int ecl_grid_get_nx( ecl_grid )")
        cls.grid.get_ny                       = cwrapper.prototype("int ecl_grid_get_ny( ecl_grid )")
        cls.grid.get_nz                       = cwrapper.prototype("int ecl_grid_get_nz( ecl_grid )")
        cls.grid.get_active                   = cwrapper.prototype("int ecl_grid_get_active_size( ecl_grid )")
        cls.grid.get_name                     = cwrapper.prototype("char* ecl_grid_get_name( ecl_grid )")
        cls.grid.get_active_index3            = cwrapper.prototype("int ecl_grid_get_active_index3( ecl_grid , int , int , int)")
        cls.grid.get_global_index3            = cwrapper.prototype("int ecl_grid_get_global_index3( ecl_grid , int , int , int)") 
        cls.grid.get_ijk1                     = cwrapper.prototype("void ecl_grid_get_ijk1( ecl_grid , int , int* , int* , int*)")
        cls.grid.get_ijk1A                    = cwrapper.prototype("void ecl_grid_get_ijk1A( ecl_grid , int , int* , int* , int*)") 
        cls.grid.get_xyz3                     = cwrapper.prototype("void ecl_grid_get_xyz3( ecl_grid , int , int , int , double* , double* , double*)")
        cls.grid.get_xyz1                     = cwrapper.prototype("void ecl_grid_get_xyz1( ecl_grid , int , double* , double* , double*)")
        cls.grid.get_xyz1A                    = cwrapper.prototype("void ecl_grid_get_xyz1A( ecl_grid , int , double* , double* , double*)")
        cls.grid.get_ijk_xyz                  = cwrapper.prototype("int  ecl_grid_get_global_index_from_xyz( ecl_grid , double , double , double , int)")
        cls.grid.num_lgr                      = cwrapper.prototype("int  ecl_grid_get_num_lgr( ecl_grid )")
        cls.grid.has_lgr                      = cwrapper.prototype("bool ecl_grid_has_lgr( ecl_grid , char* )")
        cls.grid.get_lgr                      = cwrapper.prototype("long ecl_grid_get_lgr( ecl_grid , char* )")
        cls.grid.get_cell_lgr                 = cwrapper.prototype("long ecl_grid_get_cell_lgr1( ecl_grid , int )")
        cls.grid.grid_value                   = cwrapper.prototype("double ecl_grid_get_property( ecl_grid , ecl_kw , int , int , int)")
        cls.grid.get_cell_volume              = cwrapper.prototype("double ecl_grid_get_cell_volume1( ecl_grid , int )")


        #################################################################

        cwrapper.registerType( "ecl_kw" , EclKW )
        
        #################################################################

        cwrapper.registerType( "ecl_file" , EclFile )
        cls.ecl_file.fread_alloc               = cwrapper.prototype("long   ecl_file_fread_alloc( char* )")
        cls.ecl_file.free                      = cwrapper.prototype("void   ecl_file_free( ecl_file )")
        cls.ecl_file.iget_kw                   = cwrapper.prototype("long   ecl_file_iget_kw( ecl_file , int)")
        cls.ecl_file.iget_named_kw             = cwrapper.prototype("long   ecl_file_iget_named_kw( ecl_file , char* , int)")
        cls.ecl_file.get_size                  = cwrapper.prototype("int    ecl_file_get_num_kw( ecl_file )")
        cls.ecl_file.get_unique_size           = cwrapper.prototype("int    ecl_file_get_num_distinct_kw( ecl_file )")
        cls.ecl_file.get_num_named_kw          = cwrapper.prototype("int    ecl_file_get_num_named_kw( ecl_file , char* )")
        cls.ecl_file.iget_restart_time         = cwrapper.prototype("time_t ecl_file_iget_restart_sim_date( ecl_file , int )")
        cls.ecl_file.get_restart_index         = cwrapper.prototype("int    ecl_file_get_restart_index( ecl_file , time_t)")
                                                                    
                                                                    
        
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
        
        #################################################################
        
        cls.ecl_kw.get_size                   = cwrapper.prototype("int ecl_kw_get_size( ecl_kw )")
        cls.ecl_kw.get_type                   = cwrapper.prototype("int ecl_kw_get_type( ecl_kw )")
        cls.ecl_kw.iget_char_ptr              = cwrapper.prototype("char* ecl_kw_iget_char_ptr( ecl_kw , int )")
        cls.ecl_kw.iget_bool                  = cwrapper.prototype("bool ecl_kw_iget_bool( ecl_kw , int)")
        cls.ecl_kw.iget_int                   = cwrapper.prototype("int ecl_kw_iget_int( ecl_kw , int )")
        cls.ecl_kw.iget_double                = cwrapper.prototype("double ecl_kw_iget_double( ecl_kw , int )")
        cls.ecl_kw.iget_float                 = cwrapper.prototype("float ecl_kw_iget_float( ecl_kw , int)")
        cls.ecl_kw.float_ptr                  = cwrapper.prototype("float* ecl_kw_get_float_ptr( ecl_kw )")


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
    def __init__(self , parent , c_ptr):
        self.__parent = parent   # Hold on to the parent to inhibit GC
        self.c_ptr    = c_ptr
        
    def from_param(self):
        return self.c_ptr

    @property
    def size(self):
        return Ecl.ecl_kw.get_size( self )

    @property
    def type( self ):
        __type = Ecl.ecl_kw.get_type( self )
        # enum ecl_type_enum from ecl_util.h
        if __type == ECL_CHAR_TYPE:
            return "CHAR"
        if __type == ECL_REAL_TYPE:
            return "REAL"
        if __type == ECL_DOUBLE_TYPE:
            return "DOUB"
        if __type == ECL_INT_TYPE:
            return "INTE"
        if __type == ECL_BOOL_TYPE:
            return "BOOL"
        if __type == ECL_MESS_TYPE:
            return "MESS"

        
    def iget( self , index ):
        __type = Ecl.ecl_kw.get_type( self )
        if __type == ECL_CHAR_TYPE:
            value = Ecl.ecl_kw.iget_char_ptr( self , index )

        if __type == ECL_REAL_TYPE:
            value = Ecl.ecl_kw.iget_float( self , index )

        if __type == ECL_FLOAT_TYPE:
            value = Ecl.ecl_kw.iget_double( self , index )

        if __type == ECL_INT_TYPE:
            value = Ecl.ecl_kw.iget_int( self , index )

        if __type == ECL_BOOL_TYPE:
            value = Ecl.ecl_kw.iget_bool( self , index )
            
        return value
    
    @property
    def array(self):
        type = Ecl.ecl_kw.get_type( self )
        if type == ECL_INT_TYPE:
            a = Ecl.ecl_kw.int_ptr( self )
        elif type == ECL_REAL_TYPE:
            a = Ecl.ecl_kw.float_ptr( self )
        elif type == ECL_DOUBLE_TYPE:
            a = Ecl.ecl_kw.double_ptr( self )
        else:
            a = None

        if not a == None:
            a.size       = Ecl.ecl_kw.get_size( self )
            a.__parent__ = self  # Inhibit GC
        return a
        

class EclFile:
    def __init__(self , filename):
        self.c_ptr    = Ecl.ecl_file.fread_alloc( filename )
        self.filename = filename
        
    def __del__(self):
        Ecl.ecl_file.free( self )
            
    def from_param(self):
        return self.c_ptr

    def iget_kw( self , index ):
        kw_c_ptr = Ecl.ecl_file.iget_kw( self , index )
        return EclKW( self , kw_c_ptr)
    
    def iget_named_kw( self , kw_name , index ):
        kw_c_ptr = Ecl.ecl_file.iget_named_kw( self , kw_name , index )
        return EclKW( self , kw_c_ptr)

    def restart_get( self , kw_name , dtime ):
        """
        Will lookup keyword @kw_name in the restart_file, exactly at
        time @dtime; @dtime is supposed to be a datetime.date() instance.
        """

        index = Ecl.ecl_file.get_restart_index( self , ctime( dtime ) )
        if index >= 0:
            return self.iget_named_kw( kw_name , index )
        else:
            return None

    @property
    def size(self):
        return Ecl.ecl_file.get_num_kw( self )

    @property
    def unique_size( self ):
        return Ecl.ecl_file.get_unique_size( self )

    def num_named_kw( self , kw):
        return Ecl.ecl_file.get_num_named_kw( self , kw )

    def iget_restart_time( self , index ):
        return Ecl.ecl_file.iget_restart_time( self , index )
    
    

#################################################################


class EclRFTFile:
    def __init__(self , case):
        self.c_ptr = Ecl.rft_file.load( case )

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
        cell = cls(RFT , i,j,k,pressure,depth)
        cell.__swat = swat
        cell.__sgas = sgas
        return cell

    @classmethod
    def PLTCell(cls , i,j,k,pressure,orat,grat,wrat):
        cell = cls(PLT , i,j,k,pressure,depth)
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




    
class EclGrid:
    def __init__(self , filename , lgr = None, parent = None):
        if lgr:
            self.c_ptr      = lgr
            self.data_owner = False
            self.parent     = parent     # Inhibit GC
        else:
            self.c_ptr = Ecl.grid.fread_alloc( filename )
            self.data_owner = True

            
    def __del__(self):
        if self.data_owner:
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

    def __global_index( self , active_index = None , global_index = None , ijk = None):
        set_count = 0
        if active_index:
            set_count += 1
        if global_index:
            set_count += 1
        if ijk:
            set_count += 1
            
        if not set_count == 1:
            sys.exit("The function get_xyz() requires that exactly one of the kewyord arguments active_index, global_index, ijk be set")
        
        if active_index:
            global_index = Ecl.grid.get_global_index1A( self , active_index )
        elif ijk:
            global_index = Ecl.grid.get_global_index3( self , ijk[0] , ijk[1] , ijk[2])
        
        return global_index
                 
    def get_active_index( self , ijk = None , global_index = None):
        gi = self.__global_index( global_index = global_index , ijk = ijk)
        return Ecl.grid.get_active_index1( self , gi)

    def get_global_index( self , ijk = None , active_index = None):
        gi = self.__global_index( active_index = active_index , ijk = ijk)
        return gi

    def get_ijk( self, active_index = None , global_index = None):
        i = ctypes.c_int()
        j = ctypes.c_int()
        k = ctypes.c_int()

        gi = self.__global_index( active_index = active_index , global_index = global_index)
        Ecl.grid.get_ijk1( self , gi , ctypes.byref(i) , ctypes.byref(j) , ctypes.byref(k))

        return (i.value , j.value , k.value)


    def get_xyz( self, active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)

        x = ctypes.c_double()
        y = ctypes.c_double()
        z = ctypes.c_double()

        Ecl.grid.get_xyz1( self , gi , ctypes.byref(x) , ctypes.byref(y) , ctypes.byref(z))
        
        return (x.value , y.value , z.value)



    def find_cell( self , x , y , z , start_ijk = None):
        if start_ijk:
            start_index = self.__global_index( ijk = start_ijk )
        else:
            start_index = 0
        global_index = Ecl.grid.get_ijk_xyz( self , x , y , z , start_index)

        i = ctypes.c_int()
        j = ctypes.c_int()
        k = ctypes.c_int()
        Ecl.grid.get_ijk1( self , global_index , ctypes.byref(i) , ctypes.byref(j) , ctypes.byref(k))        
        return (i.value , j.value , k.value)

    
    
    def cell_volume( self, active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)
        return Ecl.grid.get_cell_volume( self , gi)
            

    @property
    def num_lgr( self ):
        return Ecl.grid.num_lgr( self )


    def has_lgr( self , lgr_name ):
        if Ecl.grid.has_lgr( self , lgr_name ):
            return True
        else:
            return False


    def get_lgr( self , lgr_name ):
        if Ecl.grid.has_lgr(self , lgr_name ):
            lgr = EclGrid( None , lgr = Ecl.grid.get_lgr( self , lgr_name ) , parent = self)
            return lgr
        else:
            return None
        

    def get_cell_lgr( self, active_index = None , global_index = None , ijk = None):
        gi  = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)
        lgr = Ecl.grid.get_cell_lgr( self , gi )
        if lgr:
            return EclGrid( None , lgr = lgr , parent = self)
        else:
            return None


    def grid_value( self , ecl_kw , i,j,k):
        return Ecl.grid.grid_value( self , ecl_kw , i , j , k)


    
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
            self.__path = os.getwcwd()
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
            if Ecl.grid.exists( self.case ):
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



        
    
