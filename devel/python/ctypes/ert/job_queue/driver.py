#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'driver.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import ctypes
import os.path

import libjob_queue
from   job                   import Job 
from   ert.cwrap.cwrap       import *
from   ert.cwrap.cenum       import create_enum


# job_driver_type defintion from queue_driver.h
create_enum( libjob_queue.lib , "queue_driver_type_enum_iget" , "queue_driver_enum" , globals())

class Driver:
    def __init__( self , type , max_running = 1 , options = None):
        """
        Creates a new driver instance
        """
        self.c_ptr = cfunc.alloc_driver( type )
        if options:
            for (key,value) in options:
                if isinstance( value , int ):
                    cfunc.set_int_option( self , key , value )
                else:
                    cfunc.set_str_option( self , key , value )
        self.set_max_running( max_running )
        
    def is_driver_instance( self ):
        return True

    def from_param(self):
        return ctypes.c_void_p( self.c_ptr )

    def __del__( self ):
        cfunc.free_driver( self )

    def submit( self , name , cmd , run_path , argList , num_cpu = 1 , blocking = False):
        argc = len( argList )
        argv = (ctypes.c_char_p * argc)()
        argv[:] = map( str , argList )
        job_c_ptr = cfunc.submit( self , cmd , num_cpu , run_path , name , argc , argv )
        job = Job( self , job_c_ptr , blocking )
        if blocking:
            job.block()
            job = None
        return job

    def free_job( self , job ):
        cfunc.free_job( self , job )
    
    def get_status( self , job ):
        return cfunc.cget_status( self , job )
    
    def kill_job( self , job ):
        cfunc.ckill_job( self , job )

    def get_max_running( self ):
        return cfunc.get_max_running( self )
    
    def set_max_running( self , max_running ):
        cfunc.set_max_running( self , max_running )
        
    max_running = property( get_max_running , set_max_running )


class LSFDriver(Driver):
    
    def __init__(self ,
                 max_running ,
                 lsf_server = None ,
                 queue = "normal" ,
                 resource_request = None):

        # The strings should match the available keys given in the
        # lsf_driver.h header file.
        options = [("LSF_QUEUE"    , queue),
                   ("LSF_SERVER"   , lsf_server),
                   ("LSF_RESOURCE" , resource_request )]
        Driver.__init__( self , LSF_DRIVER , max_running = max_running , options = options)
        

class LocalDriver(Driver):

    def __init__( self , max_running ):
        Driver.__init__( self , LOCAL_DRIVER , max_running , options = [] )
        



class RSHDriver(Driver):

    # Changing shell to bash can come in conflict with running ssh
    # commands.
    
    def __init__( self , max_running , rsh_host_list , rsh_cmd = "/usr/bin/ssh" ):
        """
        @rsh_host_list should be a list of tuples like: (hostname , max_running) 
        """
        
        options = [("RSH_CMD" , rsh_cmd)]
        for (host,host_max) in rsh_host_list:
            options.append( ("RSH_HOST" , "%s:%d" % (host , host_max)) )
        Driver.__init__( self , RSH_DRIVER , max_running , options = options )
            



#################################################################
cwrapper = CWrapper( libjob_queue.lib )
cwrapper.registerType( "driver" , Driver ) 
cwrapper.registerType( "job"    , Job )
cfunc   = CWrapperNameSpace( "driver" )


cfunc.alloc_driver_lsf       = cwrapper.prototype("c_void_p    queue_driver_alloc_LSF( char* , char* , char* )")
cfunc.alloc_driver_local     = cwrapper.prototype("c_void_p    queue_driver_alloc_local( )")
cfunc.alloc_driver_rsh       = cwrapper.prototype("c_void_p    queue_driver_alloc_RSH( char* , c_void_p )")
cfunc.alloc_driver           = cwrapper.prototype("c_void_p    queue_driver_alloc( int )")
cfunc.set_driver_option      = cwrapper.prototype("void        queue_driver_set_option(driver , char* , char*)")


cfunc.free_driver     = cwrapper.prototype("void        queue_driver_free( driver )")
cfunc.submit          = cwrapper.prototype("c_void_p    queue_driver_submit_job( driver , char* , int , char* , char* , int , char**)")
cfunc.free_job        = cwrapper.prototype("void        queue_driver_free_job( driver , job )")
cfunc.get_status      = cwrapper.prototype("int         queue_driver_get_status( driver , job)")
cfunc.kill_job        = cwrapper.prototype("void        queue_driver_kill_job( driver , job )")
cfunc.set_str_option  = cwrapper.prototype("void        queue_driver_set_string_option( driver , char* , char*)")
cfunc.set_int_option  = cwrapper.prototype("void        queue_driver_set_int_option( driver , char* , int)")
cfunc.set_max_running = cwrapper.prototype("void        queue_driver_set_max_running( driver , int )")
cfunc.get_max_running = cwrapper.prototype("int         queue_driver_get_max_running( driver )")
