#!/usr/bin/env python

#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl.py' is part of ERT - Ensemble based Reservoir Tool. 
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

# This file contains an example installation script which can be used
# to install ERT in a central location; you should probably make your
# copy. The main reason this is so overly complex is ERTs Statoil
# heritage where it should be installed as site-wide application,
# without the use of any special priveliges. ERT as such have no
# special installation requirements; it is no requirement to use the
# approach demonstrated here.
#
# Observe that with this approach building and installing is 100%
# decoupled, i.e. you must build the distribution before you install
# it.


import os.path
from   install import File, Install


ERT_src_root = os.path.abspath( "../" )

libutil = Install( ERT_src_root )
libutil.add_file( File( "libutil/lib/libutil.a" , create_path = "lib" ) )
libutil.add_path( "libutil/include" , create_path = "include" )
libutil.add_path( "libutil/bin" , create_path = "bin" )


libecl = Install( ERT_src_root )
libecl.add_file( File( "libecl/lib/libecl.a" , create_path = "lib" ) )
libecl.add_path( "libecl/include" , create_path = "include" )
libecl.add_path( "libecl/bin" , create_path = "bin" )


libconfig = Install( ERT_src_root )
libconfig.add_file( File( "libconfig/lib/libconfig.a" , create_path = "lib" ) )
libconfig.add_path( "libconfig/include" , create_path = "include" )
libconfig.add_path( "libconfig/bin" , create_path = "bin" )


libenkf = Install( ERT_src_root )
libenkf.add_file( File( "libenkf/lib/libenkf.a" , create_path = "lib" ) )
libenkf.add_path( "libenkf/include" , create_path = "include" )
libenkf.add_path( "libenkf/applications/bin" , create_path = "bin" )


libplot = Install( ERT_src_root )
libplot.add_file( File( "libplot/lib/libplot.a" , create_path = "lib" ) )
libplot.add_path( "libplot/include" , create_path = "include" )


librms = Install( ERT_src_root )
librms.add_file( File( "librms/lib/librms.a" , create_path = "lib" ) )
librms.add_path( "librms/include" , create_path = "include" )
librms.add_path( "librms/bin" , create_path = "bin" )


libsched = Install( ERT_src_root )
libsched.add_file( File( "libsched/lib/libsched.a" , create_path = "lib" ) )
libsched.add_path( "libsched/include" , create_path = "include" )
libsched.add_path( "libsched/bin" , create_path = "bin" )


libjob_queue = Install( ERT_src_root )
libjob_queue.add_file( File( "libjob_queue/lib/libjob_queue.a" , create_path = "lib" ) )
libjob_queue.add_path( "libjob_queue/include" , create_path = "include" )
libjob_queue.add_path( "libjob_queue/bin" , create_path = "bin" )



libecl.install( target_root = "/tmp" , group = "res" , verbose = True)
libconfig.install( target_root = "/tmp" , group = "res" , verbose = True)
libutil.install( target_root = "/tmp" , group = "res" , verbose = True)
libenkf.install( target_root = "/tmp" , group = "res" , verbose = True)
libsched.install( target_root = "/tmp" , group = "res" , verbose = True)
librms.install( target_root = "/tmp" , group = "res" , verbose = True)
libplot.install( target_root = "/tmp" , group = "res" , verbose = True)
libjob_queue.install( target_root = "/tmp" , group = "res" , verbose = True)



