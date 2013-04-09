#  Copyright (C) 2012  Statoil ASA, Norway. 
#   
#  The file 'enkf_fs.py' is part of ERT - Ensemble based Reservoir Tool. 
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
from    enkf_enum             import *
from ert.ert.enums import enkf_var_type
import  libenkf
class EnkfFs(CClass):
    
    def __init__(self , c_ptr = None):
        self.owner = False
        self.c_ptr = c_ptr
        

    def __del__(self):
        if self.owner:
            cfunc.close( self )

    def has_node(self, node_key, step, member, value, type = 1):#enkf_var_type.PARAMETER):
        return cfunc.has_node(self, node_key, type, step, member, value)
    
    def fread_node(self, node, step, member, value,type = enkf_var_type.PARAMETER):
        buffer = c_buffer()
        cfunc.fread_node(self, buffer, node, type, step, member, value)

##################################################################

cwrapper = CWrapper( libenkf.lib )
cwrapper.registerType( "enkf_fs" , EnkfFs )

cfunc = CWrapperNameSpace("enkf_fs")

cfunc.close               = cwrapper.prototype("void enkf_fs_close(enkf_fs)")
cfunc.has_node            = cwrapper.prototype("bool enkf_fs_has_node(enkf_fs, char*, long, int, int, int)")
cfunc.fread_node          = cwrapper.prototype("void enkf_fs_fread_node(enkf_fs, char*, long, int, int, int)")
cfunc.get_read_dir        = cwrapper.safe_prototype("char* enkf_fs_get_read_dir(enkf_fs)")
cfunc.alloc_dirlist       = cwrapper.safe_prototype("c_void_p enkf_fs_alloc_dirlist(enkf_fs)")
