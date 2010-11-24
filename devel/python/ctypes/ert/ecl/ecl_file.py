import time
import datetime
import ctypes
import sys
import os.path
import libecl

from   ert.cwrap.cwrap       import *
from   fortio                import FortIO
from   ecl_kw                import EclKW
from   ert.util.ctime        import ctime 


class EclFile(object):

    @classmethod
    def restart_block( cls , filename , dtime = None , report_step = None):
        obj = object.__new__( cls )
        
        if dtime:
            obj.c_ptr = cfunc.restart_block_time( filename , ctime( dtime ))
        elif not report_step == None:
            obj.c_ptr = cfunc.restart_block_step( filename , report_step )
        else:
            raise TypeError("restart_block() requires either dtime or report_step argument - none given")
        
        return obj


        
    def __init__( self , filename):
        # Default initializer allocates a new instance from the C layer.
        self.c_ptr = cfunc.fread_alloc( filename )

        
    def __del__(self):
        cfunc.free( self )

    def __getitem__(self , index):
        if isinstance( index , types.IntType):
            if index < 0 or index >= cfunc.get_size( self ):
                raise IndexError
            else:
                kw_c_ptr = cfunc.iget_kw( self , index )
                return EclKW.ref( kw_c_ptr , self )
        else:
            raise TypeError

    def from_param(self):
        return self.c_ptr


    def iget_kw( self , index ):
        return self.__getitem__( index )
    
    def iget_named_kw( self , kw_name , index ):
        kw_c_ptr = cfunc.iget_named_kw( self , kw_name , index )
        return EclKW.ref( kw_c_ptr , self )

    def restart_get_kw( self , kw_name , dtime ):
        """
        Will lookup keyword @kw_name in the restart_file, exactly at
        time @dtime; @dtime is supposed to be a datetime.date() instance.
        """

        index = cfunc.get_restart_index( self , ctime( dtime ) )
        if index >= 0:
            return self.iget_named_kw( kw_name , index )
        else:
            return None

    @property
    def size(self):
        return cfunc.get_size( self )

    @property
    def unique_size( self ):
        return cfunc.get_unique_size( self )

    def num_named_kw( self , kw):
        return cfunc.get_num_named_kw( self , kw )

    def has_kw( self , kw , num = 0):
        num_named_kw = self.num_named_kw( kw )
        if num_named_kw > num:
            return True
        else:
            return False

    
    def iget_restart_time( self , index ):
        return cfunc.iget_restart_time( self , index )
    
    @property
    def name(self):
        return cfunc.get_src_file( self )
    


# 2. Creating a wrapper object around the libecl library, 
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_file" , EclFile )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_file")

cfunc.fread_alloc               = cwrapper.prototype("long   ecl_file_fread_alloc( char* )")
cfunc.new                       = cwrapper.prototype("long   ecl_file_alloc_empty(  )")
cfunc.restart_block_time        = cwrapper.prototype("long   ecl_file_fread_alloc_unrst_section_time( char* , time_t )")
cfunc.restart_block_step        = cwrapper.prototype("long   ecl_file_fread_alloc_unrst_section( char* , int )")
cfunc.iget_kw                   = cwrapper.prototype("long   ecl_file_iget_kw( ecl_file , int)")
cfunc.iget_named_kw             = cwrapper.prototype("long   ecl_file_iget_named_kw( ecl_file , char* , int)")
cfunc.free                      = cwrapper.prototype("void       ecl_file_free( ecl_file )")
cfunc.get_size                  = cwrapper.prototype("int        ecl_file_get_num_kw( ecl_file )")
cfunc.get_unique_size           = cwrapper.prototype("int        ecl_file_get_num_distinct_kw( ecl_file )")
cfunc.get_num_named_kw          = cwrapper.prototype("int        ecl_file_get_num_named_kw( ecl_file , char* )")
cfunc.iget_restart_time         = cwrapper.prototype("time_t     ecl_file_iget_restart_sim_date( ecl_file , int )")
cfunc.get_restart_index         = cwrapper.prototype("int        ecl_file_get_restart_index( ecl_file , time_t)")
cfunc.insert_kw                 = cwrapper.prototype("void       ecl_file_insert_kw( ecl_file , ecl_kw , bool , char* , int )")
cfunc.del_kw                    = cwrapper.prototype("void       ecl_file_delete_kw( ecl_file , char* , int)")
cfunc.get_src_file              = cwrapper.prototype("char*      ecl_file_get_src_file( ecl_file )")

