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
init files, grid files, summary files and RFT files.

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
import types
import libecl

from   ert.cwrap.cwrap       import *
from   ecl_kw                import EclKW
from   ert.util.ctime        import ctime 
from   ert.util.stringlist   import StringList

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
        will return None.
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
            obj = None

        return obj


    @classmethod
    def contains_report_step( cls , filename , report_step ):
        """
        Will check if the @filename contains @report_step.

        This classmethod works by opening the file @filename and
        searching through linearly to see if an ecl_kw with value
        corresponding to @report_step can be found. Since this is a
        classmethod it is invoked like this:

           import ert.ecl.ecl as ecl
           ....
           if ecl.EclFile.contains_report_step("ECLIPSE.UNRST" , 20):
              print "OK - file contains report step 20"
           else:
              print "File does not contain report step 20"

        If you have already loaded the file into an EclFile instance
        you should use the has_report_step() method instead.
        """
        return cfunc.contains_report_step( filename , report_step )

    
    @classmethod
    def contains_sim_time( cls , filename , dtime ):
        """
        Will check if the @filename contains simulation at @dtime.

        This classmethod works by opening the file @filename and
        searching through linearly to see if a result block at the
        time corresponding to @dtime can be found. Since this is a
        classmethod it is invoked like this:

           import ert.ecl.ecl as ecl
           ....
           if ecl.EclFile.contains_sim_time("ECLIPSE.UNRST" , datetime.datetime( 2007 , 10 , 10) ):
              print "OK - file contains 10th of October 2007"
           else:
              print "File does not contain 10th of October 2007"

        If you have already loaded the file into an EclFile instance
        you should use the has_sim_time() method instead.
        """
        return cfunc.contains_sim_time( filename , ctime(dtime) )


    
    @classmethod
    def file_report_steps( cls , filename ):
        """
        Will identify the available report_steps from @filename.
        """
        report_steps = []
        f = cls( filename , kw_list = ["SEQNUM"] )
        for s in f:
            report_steps.append( s[0] )
        return report_steps

        
    def __init__( self , filename ):
        """
        Loads the complete file @filename.

        Will create a new EclFile instance with the content of file
        @filename. The file @filename must be in 'restart format' -
        otherwise it will be crash and burn. 
        
        The optional argument @kw_list can be used to limit the
        loading to only some of the keywords in the file, @kw_list
        should be a an ordinary Python list of strings. To load only
        the solution data from a restart file:

            sol_data = ecl.EclFile("ECLIPSE.UNRST" , kw_list = ["PRESSURE" , "SWAT" , "SGAS"])

        When the file has been loaded the EclFile instance can be used
        to query for and get reference to the EclKW instances
        constituting the file, like e.g. SWAT from a restart file or
        FIPNUM from an INIT file.
        """
        self.c_ptr = cfunc.open( filename )
        
        
    def __del__(self):
        if self.c_ptr:
            cfunc.close( self )

    def restart_section( self, report_step = None , sim_time = None):
        """
        Will create a new EclFile instance consisting of one restart block.

        You must specify either a report step with the @report_step
        argument, or a true time with the @sim_time argument; if
        neither argument is given exception TypeError wiull be
        raised. If present the @sim_time argument should be a datetime
        instance.
        
        If the restart section you ask for can not be found the method
        will return a EclFile instance which evaluates to False;
        alternatively you can query first with the has_report_step()
        or has_sim_time() methods.

        This method should be used when you have already loaded the
        complete file; if you only want to load a section from the
        file you can use the classmethod restart_block().
        """
        if report_step:
            c_ptr = cfunc.copy_restart_section( self , report_step )
        elif sim_time:
            c_ptr = cfunc.copy_restart_section_time_t( self , ctime( sim_time ) )
        else:
            raise TypeError("restart_section() requires either dtime or report_step argument - none given")
        
        if c_ptr:
            obj = object.__new__( EclFile )
            obj.c_ptr = c_ptr
        else:
            obj = None

        return obj
        

    def __getitem__(self , index):
        """
        Implements [] operator; index can be integer or key.

        Will look up EclKW instances from the current EclFile
        instance. The @index argument can either be an integer, in
        which case the method will return EclKW number @index, or
        alternatively a keyword string, in which case the method will
        return a list of EclKW instances with that keyword:

           restart_file = ecl_file.EclFile("ECLIPSE.UNRST")
           kw9 = restart_file[9]
           swat_list = restart_file["SWAT"]

        The keyword based lookup can be combined with an extra [] to
        get EclKW instance nr:

           swat9 = restart_file["SWAT"][9]

        Will return the 10'th SWAT keyword from the restart file. The
        following example will iterate over all the SWAT keywords in a
        restart file:

           restart_file = ecl_file.EclFile("ECLIPSE.UNRST")
           for swat in restart_file["SWAT"]:
               ....
        """

        if isinstance( index , types.IntType):
            if index < 0 or index >= cfunc.get_size( self ):
                raise IndexError
            else:
                kw_c_ptr = cfunc.iget_kw( self , index )
                return EclKW.ref( kw_c_ptr , self )
        else:
            if isinstance( index , types.StringType):
                kw_index = index
                kw_list = []
                for index in range( self.num_named_kw( kw_index )):
                    kw_list.append(  self.iget_named_kw( kw_index , index))
                return kw_list
            else:
                raise TypeError("Index must be integer or string (keyword)")
        
    
    @classmethod
    def from_param( cls , obj ):
        if obj is None:
            return ctypes.c_void_p()
        else:
            return obj.c_ptr

    def iget_kw( self , index , copy = False):
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
        kw = self[index]
        if copy:
            return EclKW.copy( kw )
        else:
            return kw
  
  
    def iget_named_kw( self , kw_name , index , copy = False):
        """
        Will return EclKW nr @index reference with header @kw_name.
        
        The keywords in a an ECLIPSE file are organized in a long
        linear array; keywords with the same name can occur many
        times. For instance a summary data[1] file might look like
        this:

           SEQHDR  
           MINISTEP
           PARAMS  
           MINISTEP
           PARAMS
           MINISTEP
           PARAMS
           ....

        To get the third 'PARAMS' keyword you can use the method call:
  
            params_kw = file.iget_named_kw( "PARAMS" , 2 )

        The functionality of the iget_named_kw() method is also
        available through the __getitem__() method as:
        
           params_kw = file["PARAMS"][2]

        Observe that the returned EclKW instance is only a reference
        to the data owned by the EclFile instance.
        
        [1]: For working with summary data you are probably better off
             using the EclSum class.
        """
        kw_c_ptr = cfunc.iget_named_kw( self , kw_name , index )
        ecl_kw = EclKW.ref( kw_c_ptr , self )
        if copy:
            return EclKW.copy( ecl_kw )
        else:
            return ecl_kw


    def restart_get_kw( self , kw_name , dtime , copy = False):
        """
        Will return EclKW @kw_name from restart file at time @dtime.

        This function assumes that the current EclFile instance
        represents a restart file. It will then look for keyword
        @kw_name exactly at the time @dtime; @dtime is a datetime
        instance:

            file = EclFile( "ECLIPSE.UNRST" )
            swat2010 = file.restart_get_kw( "SWAT" , datetime.datetime( 2000 , 1 , 1 ))

        By default the returned kw instance is a reference to the
        ecl_kw still contained in the EclFile instance; i.e. the kw
        will become a dangling reference if the EclFile instance goes
        out of scope. If the optional argument @copy is True the
        returned kw will be a true copy.

        If the file does not have the keyword at the specified time the
        function will return None.
        """
        index = cfunc.get_restart_index( self , ctime( dtime ) )
        if index >= 0:
            kw = self.iget_named_kw( kw_name , index )
            if copy:
                return EclKW.copy( kw )
            else:
                return kw
        else:
            return None


    def replace_kw( self , old_kw , new_kw):
        """
        Will replace @old_kw with @new_kw in current EclFile instance.

        This method can be used to replace one of the EclKW instances
        in the current EclFile. The @old_kw reference must be to the
        actual EclKW instance in the current EclFile instance (the
        final comparison is based on C pointer equality!), i.e. it
        must be a reference (not a copy) from one of the ??get_kw??
        methods of the EclFile class. In the example below we replace
        the SWAT keyword from a restart file:

           swat = file.iget_named_kw( "SWAT" , 0 )
           new_swat = swat * 0.25
           file.replace_kw( swat , new_swat )


        The C-level ecl_file_type structure takes full ownership of
        all installed ecl_kw instances; mixing the garbage collector
        into it means that this is quite low level - and potentially
        dangerous!
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
        """
        The number of keywords in the current EclFile object.
        """
        return cfunc.get_size( self )

    @property
    def unique_size( self ):
        """
        The number of unique keyword (names) in the current EclFile object.
        """
        return cfunc.get_unique_size( self )

    
    @property
    def headers(self):
        """
        Will return a list of the headers of all the keywords.
        """
        header_list = []
        for index in range(self.size):
            kw = self[index]
            header_list.append( kw.header )
        return header_list
    
    @property
    def report_steps( self ):
        """
        Will return a list of all report steps.

        The method works by iterating through the whole restart file
        looking for 'SEQNUM' keywords; if the current EclFile instance
        is not a restart file it will not contain any 'SEQNUM'
        keywords and the method will simply return an empty list.
        """
        steps = []
        seqnum_list = self["SEQNUM"]
        for kw in self["SEQNUM"]:
            steps.append( kw[0] )
        return steps
    
    @property
    def report_dates( self ):
        """
        Will return a list of the dates for all report steps.
        
        The method works by iterating through the whole restart file
        looking for 'INTEHEAD' keywords; the method can probably be
        tricked by other file types also containing an INTEHEAD
        keyword.
        """
        dates = []
        for index in range( self.num_named_kw( 'INTEHEAD' )):
            dates.append( self.iget_restart_sim_time( index ))
        return dates
    

    def num_named_kw( self , kw):
        """
        The number of keywords with name == @kw in the current EclFile object.
        """
        return cfunc.get_num_named_kw( self , kw )

    def has_kw( self , kw , num = 0):
        """
        Check if current EclFile instance has a keyword @kw.

        If the optional argument @num is given it will check if the
        EclFile has at least @num occurences of @kw.
        """
        num_named_kw = self.num_named_kw( kw )
        if num_named_kw > num:
            return True
        else:
            return False

    def has_report_step( self , report_step ):
        """
        Checks if the current EclFile has report step @report_step.

        If the EclFile in question is not a restart file, you will
        just get False. If you want to check if the file contains the
        actual report_step before loading the file, you should use the
        classmethod contains_report_step() instead.
        """
        return cfunc.has_report_step( self , report_step )

    
    def has_sim_time( self , dtime ):
        """
        Checks if the current EclFile has data for time @dtime.

        The implementation goes through all the INTEHEAD headers in
        the EclFile, i.e. it can be fooled (and probably crash and
        burn) if the EclFile instance in question is has INTEHEAD
        keyword(s), but is still not a restart file. The @dtime
        argument should be a normal python datetime instance.
        """
        return cfunc.has_sim_time( self , ctime(dtime) )    

    
    def iget_restart_sim_time( self , index ):
        """
        Will locate restart block nr @index and return the true time
        as a datetime instance.
        """
        ctime = cfunc.iget_restart_time( self , index ) 
        return ctime.datetime()


    @property
    def name(self):
        """
        Name of the file currently loaded.
        """
        return cfunc.get_src_file( self )
    
    def fwrite( self , fortio ):
        """
        Will write current EclFile instance to fortio stream.

        ECLIPSE is written in Fortran; and a "special" handle for
        Fortran IO must be used when reading and writing these files.
        This method will write the current EclFile instance to a
        FortIO stream already opened for writing:

           import ert.ecl.ecl as ecl
           ...
           fortio = ecl.FortIO( "FILE.XX" )
           file.fwrite( fortio )
           fortio.close()
           
        """
        cfunc.fwrite( self , fortio , 0 )

        

# 2. Creating a wrapper object around the libecl library, 
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_file" , EclFile )


# 3. Installing the c-functions used to manipulate ecl_file instances.
#    These functions are used when implementing the EclFile class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_file")

cfunc.open                        = cwrapper.prototype("c_void_p    ecl_file_open( char* )")
cfunc.new                         = cwrapper.prototype("c_void_p    ecl_file_alloc_empty(  )")
cfunc.restart_block_time          = cwrapper.prototype("c_void_p    ecl_file_fread_alloc_unrst_section_time( char* , time_t )")
cfunc.restart_block_step          = cwrapper.prototype("c_void_p    ecl_file_fread_alloc_unrst_section( char* , int )")
cfunc.iget_kw                     = cwrapper.prototype("c_void_p    ecl_file_iget_kw( ecl_file , int)")
cfunc.iget_named_kw               = cwrapper.prototype("c_void_p    ecl_file_iget_named_kw( ecl_file , char* , int)")
cfunc.close                       = cwrapper.prototype("void        ecl_file_close( ecl_file )")
cfunc.get_size                    = cwrapper.prototype("int         ecl_file_get_size( ecl_file )")
cfunc.get_unique_size             = cwrapper.prototype("int         ecl_file_get_num_distinct_kw( ecl_file )")
cfunc.get_num_named_kw            = cwrapper.prototype("int         ecl_file_get_num_named_kw( ecl_file , char* )")
cfunc.iget_restart_time           = cwrapper.prototype("time_t      ecl_file_iget_restart_sim_date( ecl_file , int )")
cfunc.get_restart_index           = cwrapper.prototype("int         ecl_file_get_restart_index( ecl_file , time_t)")
cfunc.get_src_file                = cwrapper.prototype("char*       ecl_file_get_src_file( ecl_file )")
cfunc.replace_kw                  = cwrapper.prototype("void        ecl_file_replace_kw( ecl_file , ecl_kw , ecl_kw , bool)")
cfunc.fwrite                      = cwrapper.prototype("void        ecl_file_fwrite_fortio( ecl_file , fortio , int)")
cfunc.has_instance                = cwrapper.prototype("bool        ecl_file_has_kw_ptr(ecl_file , ecl_kw)")
cfunc.has_report_step             = cwrapper.prototype("bool        ecl_file_has_report_step( ecl_file , int)")
cfunc.has_sim_time                = cwrapper.prototype("bool        ecl_file_has_sim_time( ecl_file , time_t )")
cfunc.contains_report_step        = cwrapper.prototype("bool        ecl_file_contains_report_step( char* , int )")
cfunc.contains_sim_time           = cwrapper.prototype("bool        ecl_file_contains_sim_time( char* , time_t )")
cfunc.copy_restart_section        = cwrapper.prototype("c_void_p    ecl_file_copy_restart_section( ecl_file , int )")
cfunc.copy_restart_section_time_t = cwrapper.prototype("c_void_p    ecl_file_copy_restart_section_time_t( ecl_file , time_t )")

