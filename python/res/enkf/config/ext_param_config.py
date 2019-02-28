#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'ext_param_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res import ResPrototype
from ecl.util.util import StringList


class ExtParamConfig(BaseCClass):
    TYPE_NAME = "ext_param_config"
    _alloc     = ResPrototype("void*   ext_param_config_alloc( char*, stringlist )", bind = False)
    _size      = ResPrototype("int     ext_param_config_get_data_size( ext_param_config )")
    _iget_key  = ResPrototype("char*   ext_param_config_iget_key( ext_param_config , int)")
    _free      = ResPrototype("void    ext_param_config_free( ext_param_config )")
    _has_key   = ResPrototype("bool    ext_param_config_has_key( ext_param_config , char* )")

    def __init__(self, key, input_keys):
        keys = StringList( initial = input_keys )
        c_ptr = self._alloc(key, keys)
        super(ExtParamConfig, self).__init__(c_ptr)

    def __len__(self):
        return self._size( )

    def __contains__(self, key):
        return self._has_key( key )

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index >= len(self):
            raise IndexError("Invalid index:%d" % index)

        return self._iget_key( index )


    def keys(self):
        index = 0
        while index < len(self):
            yield self[index]
            index += 1


    def free(self):
        self._free( )
