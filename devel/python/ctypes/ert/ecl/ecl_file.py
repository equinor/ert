#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_file.py' is part of ERT - Ensemble based Reservoir Tool. 
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
"""
The ecl_file module contains functionality to load a an ECLIPSE file
in 'restart format'. Files of 'restart format' include restart files,
init file, grid files, summary files and RFT files.

The ecl_file implementation is agnostic[1] to the content and
structure of the file; more specialized classes like EclSum and
EclGrid use the EclFile functionality for low level file loading.

The typical usage involves loading a complete file, and then
subsequently querying for various keywords. In the example below we
load a restart file, and ask for the SWAT keyword:

   file = EclFile( "ECLIPSE.X0067" )
   swat_kw = file.iget_named_kw( "SWAT" , 0 )

The ecl_file module is a thin wrapper around the ecl_file.c
implementation from the libecl library.

[1]: In particular for restart files, which do not have a special
     RestartFile class, there is some specialized functionality.
"""

import datetime
import ctypes
import libecl

from   ert.cwrap.cwrap       import *
from   ecl_kw                import EclKW
from   ert.util.ctime        import ctime 


class EclFile(object):

    @classmethod
    def restart_block( cls , filename , dtime = None , report_step = None):
        """
        Load one report step from unified restart file.

        Unified restart files can be prohibitively large; with this
        class method it is possible to load only one report
        step. Which report step you are interested in must be
        specified with either one of the optional arguments
        @report_step or @dtime. If present @dtime should be a normal
        python datetime instance:

            block1 = EclFile.restart_block( "ECLIPSE.UNRST" , dtime = datetime.datetime( year , month , day ))
            block2 = EclFile.restart_block( "ECLIPSE.UNRST" , report_step = 67 )

        If the block you are asking for can not be found the method
        will return a False value.
        """
        if dtime:
            c_ptr = cfunc.restart_block_time( filename , ctime( dtime ))
        elif not report_step == None:
            c_ptr = cfunc.restart_block_step( filename , report_step )
        else:
            raise TypeError("restart_block() requires either dtime or report_step argument - none given")
        
        if c_ptr:
            obj = object.__new__( cls )
            obj.c_ptr = c_ptr
        else:
            obj = EclFile.NULL()

        return obj


    @classmethod
    def NULL( cls ):
        obj = object.__new__( cls )
        obj.c_ptr  = None 
        obj.parent = None
        obj.data_owner = False
        return obj


        
    def __init__( self , filename):
        """
        Loads the complete file @filename.

        Will create a new EclFile instance with the content of file
        @filename. The file @filename must be in 'restart format' -
        otherwise it will be crash and burn. 
        
        When the file has been loaded the EclFile instance can be used
        to query for and get reference to different EclKW instances
        like e.g. SWAT from a restart file or FIPNUM from an INIT
        file.
        """
        self.c_ptr = cfunc.fread_alloc( filename )

        
    def __del__(self):
        if self.c_ptr:
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
        

    def __nonzero__(self):
        if self.c_ptr:
            return True
        else:
            return False

    def from_param(self):
        """
        ctypes utility function.
        
        ctypes utility method facilitating transparent mapping between
        python EclFile instances and C based ecl_file_type pointers.
        """
        return self.c_ptr


    def iget_kw( self , index ):
        """
        Will return EclKW instance nr @index.
        
        In the files loaded with the EclFile implementation the
        ECLIPSE keywords come sequentially in a long series, an INIT
        file might have the following keywords:

          INTEHEAD
          LOGIHEAD
          DOUBHEAD
          PORV    
          DX      
          DY      
          DZ      
          PERMX   
          PERMY   
          PERMZ   
          MULTX   
          MULTY   
          .....
          
        The iget_kw() method will give you a EclKW reference to
        keyword nr @index. This functionality is also available
        through the index operator []:

           file = EclFile( "ECLIPSE.INIT" )
           permx = file.iget_kw( 7 )
           permz = file[ 9 ]

        Observe that the returned EclKW instance is only a reference
        to the data owned by the EclFile instance.

        The method iget_named_kw() which lets you specify the name of
        the keyword you are interested in is in general more useful
        than this method.
        """
        return self.__getitem__( index )
  
  
    def iget_named_kw( self , kw_name , index ):
        """
        Will return EclKW nr @index reference with header @kw_name.
        
        The keywords in a an ECLIPSE file are organized in a long
        linear array; keywords with the same name can occur many
        times. For instance a summary data[1] file might look like this:

           SEQHDR  
           MINISTEP
           PARAMS  
           MINISTEP
           PARAMS
           MINISTEP
           PARAMS
           ....

        To get the third 'PARAMS' keyword you can use the method call:
  
            params_kw = iget_named_kw( "PARAMS" , 2 )
            
        Observe that the returned EclKW instance is only a reference
        to the data owned by the EclFile instance.
        
        [1]: For working with summary data you are probably better off
             using the EclSum class.
        """

        kw_c_ptr = cfunc.iget_named_kw( self , kw_name , index )
        return EclKW.ref( kw_c_ptr , self )


    def restart_get_kw( self , kw_name , dtime ):
        """
        Will return EclKW @kw_name from restart file at time @dtime.

        This function assumes that the current EclFile instance
        represents a restart file. It will then look for keyword
        @kw_name exactly at the time @dtime; @dtime is a datetime
        instance:

            file = EclFile( "ECLIPSE.UNRST" )
            swat2010 = file.restart_get_kw( "SWAT" , datetime.datetime( 2000 , 1 , 1 ))

        If the file does not have the keyword at the specified time
        the function will return None.
        """
        
        index = cfunc.get_restart_index( self , ctime( dtime ) )
        if index >= 0:
            return self.iget_named_kw( kw_name , index )
        else:
            return None


    def replace_kw( self , old_kw , new_kw):
        """
        Will replace the one keyword in current EclFile instance.

        This method can be used to replace one of the EclKW instances
        in the current EclFile. The @old_kw reference must be to the
        actual EclKW instance in the current EclFile instance (the
        final comparison is based on C pointer equality!), i.e. it
        must come from one of the ??get_kw?? methods of the EclFile
        class. In the example below we replace the SWAT keyword from a
        restart file:

           swat = file.iget_named_kw( "SWAT" , 0 )
           new_swat = swat * 0.25
           file.replace_kw( swat , new_swat )


        The C-level ecl_file_type structure takes full ownership of
        all installed ecl_kw instances; mixing the garbage collector
        into it means that this is quite low level.
        """
        # We ensure that this scope owns the new_kw instance; the
        # new_kw will be handed over to the ecl_file instance, and we
        # can not give away something we do not alreeady own.
        if not new_kw.data_owner:
            new_kw = EclKW.copy( new_kw )

        # The ecl_file instance will take responsability for freeing
        # this ecl_kw instance.
        new_kw.data_owner = False
        cfunc.replace_kw( self , old_kw , new_kw , False )



    @property
    def size(self):
        """The number of keywords in the current EclFile object."""
        return cfunc.get_size( self )

    @property
    def unique_size( self ):
        """The number of unique keyword (names) in the current EclFile object."""
        return cfunc.get_unique_size( self )

    def num_named_kw( self , kw):
        """The number of keywords with name == @kw in the current EclFile object."""
        return cfunc.get_num_named_kw( self , kw )

    def has_kw( self , kw , num = 0):
        num_named_kw = self.num_named_kw( kw )
        if num_named_kw > num:
            return True
        else:
            return False

    def iget_restart_sim_time( self , index ):
        return cfunc.iget_restart_time( self , index )

    @property
    def name(self):
        return cfunc.get_src_file( self )
    
    def fwrite( self , fortio ):
        cfunc.fwrite( self , fortio , 0 )



# 2. Creating a wrapper object around the libecl library, 
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_file" , EclFile )


# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_file")

cfunc.fread_alloc               = cwrapper.prototype("c_void_p    ecl_file_fread_alloc( char* )")
cfunc.new                       = cwrapper.prototype("c_void_p    ecl_file_alloc_empty(  )")
cfunc.restart_block_time        = cwrapper.prototype("c_void_p    ecl_file_fread_alloc_unrst_section_time( char* , time_t )")
cfunc.restart_block_step        = cwrapper.prototype("c_void_p    ecl_file_fread_alloc_unrst_section( char* , int )")
cfunc.iget_kw                   = cwrapper.prototype("c_void_p    ecl_file_iget_kw( ecl_file , int)")
cfunc.iget_named_kw             = cwrapper.prototype("c_void_p    ecl_file_iget_named_kw( ecl_file , char* , int)")
cfunc.free                      = cwrapper.prototype("void        ecl_file_free( ecl_file )")
cfunc.get_size                  = cwrapper.prototype("int         ecl_file_get_num_kw( ecl_file )")
cfunc.get_unique_size           = cwrapper.prototype("int         ecl_file_get_num_distinct_kw( ecl_file )")
cfunc.get_num_named_kw          = cwrapper.prototype("int         ecl_file_get_num_named_kw( ecl_file , char* )")
cfunc.iget_restart_time         = cwrapper.prototype("time_t      ecl_file_iget_restart_sim_date( ecl_file , int )")
cfunc.get_restart_index         = cwrapper.prototype("int         ecl_file_get_restart_index( ecl_file , time_t)")
cfunc.insert_kw                 = cwrapper.prototype("void        ecl_file_insert_kw( ecl_file , ecl_kw , bool , char* , int )")
cfunc.del_kw                    = cwrapper.prototype("void        ecl_file_delete_kw( ecl_file , char* , int)")
cfunc.get_src_file              = cwrapper.prototype("char*       ecl_file_get_src_file( ecl_file )")
cfunc.replace_kw                = cwrapper.prototype("void        ecl_file_replace_kw( ecl_file , ecl_kw , ecl_kw , bool)")
cfunc.fwrite                    = cwrapper.prototype("void        ecl_file_fwrite_fortio( ecl_file , fortio , int)")
