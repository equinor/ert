#  Copyright (C) 2013  Statoil ASA, Norway. 
#   
#  The file 'plot_config.py' is part of ERT - Ensemble based Reservoir Tool. 
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

import  ctypes
from    ert.cwrap.cwrap       import *
from    ert.cwrap.cclass      import CClass
from    ert.util.tvector      import * 
import  libplot
class PlotDriver(CClass):
    def __init__(self , c_ptr = None):
        self.owner = False
        self.c_ptr = c_ptr
        
        
    def __del__(self):
        if self.owner:
            cfunc.free( self )
  ##################################################################

cwrapper = CWrapper( libplot.lib )
cwrapper.registerType( "plot_driver" , PlotDriver )

cfunc = CWrapperNameSpace("plot_driver")
##################################################################
##################################################################
cfunc.free                = cwrapper.prototype("void plot_driver_free( plot_driver )")          
