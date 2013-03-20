#  Copyright (C) 2012  Statoil ASA, Norway. 
#   
#  The file 'enkf_config_node.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import  libenkf
from    ert.enkf.gen_data_config import GenDataConfig
from    ert.enkf.gen_kw_config   import GenKwConfig
from    ert.enkf.field_config    import FieldConfig

class EnkfConfigNode(CClass):
    
    def __init__(self , c_ptr = None):
        self.owner = False
        self.c_ptr = c_ptr
        
        
    def __del__(self):
        if self.owner:
            cfunc.free( self )

    @property
    def get_impl_type( self ):
        return cfunc.get_impl_type( self )

    @property
    def get_ref(self):
        return cfunc.get_ref( self)

    @property
    def get_min_std_file(self):
        return cfunc.get_min_std_file( self)

    @property
    def get_enkf_outfile(self):
        return cfunc.get_enkf_outfile( self)

    @property
    def field_model(self):
        obj = FieldConfig()
        obj.c_ptr = cfunc.get_ref( self)
        return obj

    @property
    def data_model(self):
        obj = GenDataConfig()
        obj.c_ptr = cfunc.get_ref( self)
        return obj

    @property
    def keyword_model(self):
        obj = GenKwConfig()
        obj.c_ptr = cfunc.get_ref( self)
        return obj

    @property
    def get_enkf_infile(self):
        return cfunc.get_enkf_infile( self)
##################################################################

cwrapper = CWrapper( libenkf.lib )
cwrapper.registerType( "enkf_config_node" , EnkfConfigNode )

cfunc = CWrapperNameSpace("enkf_config_node")
##################################################################
##################################################################
cfunc.free                = cwrapper.prototype("void enkf_config_node_free( enkf_config_node )")
cfunc.iget_sim_days       = cwrapper.safe_prototype("double enkf_config_node_iget_sim_days(enkf_config_node, int, int)")
cfunc.iget_sim_time       = cwrapper.safe_prototype("time_t enkf_config_node_iget_sim_time(enkf_config_node, int, int)")
cfunc.get_ref             = cwrapper.prototype("c_void_p enkf_config_node_get_ref(enkf_config_node)")
cfunc.get_impl_type       = cwrapper.prototype("c_void_p enkf_config_node_get_impl_type(enkf_config_node)")
cfunc.get_enkf_outfile    = cwrapper.prototype("char* enkf_config_node_get_enkf_outfile(enkf_config_node)")
cfunc.get_min_std_file    = cwrapper.prototype("char* enkf_config_node_get_min_std_file(enkf_config_node)")
cfunc.get_enkf_infile    = cwrapper.prototype("char* enkf_config_node_get_enkf_infile(enkf_config_node)")
