#  Copyright (C) 2012  Statoil ASA, Norway.
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
from cwrap import BaseCClass
from ert.config import SchemaItem
from ert.enkf import EnkfPrototype

class PlotSettings(BaseCClass):
    TYPE_NAME    = "plot_settings"
    _alloc       = EnkfPrototype("void*  plot_settings_alloc( )", bind = False)
    _free        = EnkfPrototype("void  plot_settings_free( plot_settings )")
    _get_path    = EnkfPrototype("char* plot_settings_get_path(plot_settings)")
    _set_path    = EnkfPrototype("void  plot_settings_set_path(plot_settings, char*)")
    _alloc_keys  = EnkfPrototype("stringlist_obj plot_settings_alloc_keys(plot_settings)")
    _has_key     = EnkfPrototype("bool plot_settings_has_key(plot_settings, char*)")
    _set         = EnkfPrototype("bool plot_settings_set_value(plot_settings, char*, char*)")
    _get         = EnkfPrototype("char* plot_settings_get_value(plot_settings, char*)")
    _get_type    = EnkfPrototype("config_content_type_enum plot_settings_get_value_type(plot_settings, char*)")
    
    
    def __init__(self):
        c_ptr = self._alloc( )
        super(PlotSettings, self).__init__(c_ptr)


    def __setitem__(self,key,value):
        if key in self:
            set_ok = self._set( key , str(value))
            if not set_ok:
                raise TypeError("Setting %s=%s failed \n" % (key , value))
        else:
            raise KeyError("PlotSettings object does not support key:%s" % key)

        
    def __getitem__(self,key):
        if key in self:
            string_value = self._get( key )
            return SchemaItem.convert( self._get_type( key ) , string_value )
        else:
            raise KeyError("PlotSettings object does not support key:%s" % key)

        
    def __contains__(self , key):
        return self._has_key( key )

    
    def getPath(self):
        """ @rtype: str """
        return self._get_path()

    def setPath(self, path):
        self._set_path(path)

    def free(self):
        self._free()

    def keys(self):
        return self._alloc_keys( )
