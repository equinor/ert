#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'ext_param.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os.path

from cwrap import BaseCClass, CFILE

from res import ResPrototype
from res.enkf.config import ExtParamConfig


class ExtParam(BaseCClass):
    TYPE_NAME = "ext_param"
    _alloc    = ResPrototype("void*  ext_param_alloc( ext_param_config )", bind = False)
    _free     = ResPrototype("void   ext_param_free( ext_param )")
    _size     = ResPrototype("int    ext_param_get_size( ext_param )")
    _has_key  = ResPrototype("bool   ext_param_has_key( ext_param , char*)")
    _iset     = ResPrototype("void   ext_param_iset( ext_param, int, double)")
    _key_set  = ResPrototype("void   ext_param_key_set( ext_param, char*, double)")
    _iget     = ResPrototype("double ext_param_iget( ext_param, int)")
    _key_get  = ResPrototype("double ext_param_key_get( ext_param, char*)")
    _iget_key = ResPrototype("char*  ext_param_iget_key( ext_param, int)")
    _export   = ResPrototype("void   ext_param_json_export( ext_param, char*)")

    def __init__(self, config):
        c_ptr = self._alloc( config )
        self.config = config
        super(ExtParam, self).__init__(c_ptr)


    def __contains__(self, key):
        return self._has_key( key )

    def __len__(self):
        return self._size( )


    def __getitem__(self,index):
        if isinstance(index,int):
            if index < 0:
                index += len(self)

            if index >= len(self):
                raise IndexError("Invalid index:%d" % index)
            return self._iget( index )
        else:
            if not index in self:
                raise KeyError("No such key:%s" % index)
            return self._key_get( index )


    def __setitem__(self,index,value):
        if isinstance(index,int):
            if index >= len(self):
                raise IndexError("Invalid index:%d" % index)
            self._iset( index, value )
        else:
            if not index in self:
                raise KeyError("No such key:%s" % index)
            self._key_set( index, value )


    # This could in the future be specialized to take a numpy vector,
    # which could be vector-assigned in C.
    def set_vector(self, values):
        if len(values) != len(self):
            raise ValueError("Size mismatch")

        for index,value in enumerate(values):
            self[index] = value


    def keys(self):
        index = 0
        while index < len(self):
            yield self._iget_key( index )
            index += 1


    def free(self):
        self._free( )


    def export(self, fname):
        self._export( fname )
