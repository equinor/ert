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
from six import string_types

from res import ResPrototype
from res.enkf.config import ExtParamConfig


class ExtParam(BaseCClass):
    TYPE_NAME = "ext_param"
    _alloc = ResPrototype("void*  ext_param_alloc( ext_param_config )", bind=False)
    _free = ResPrototype("void   ext_param_free( ext_param )")
    _iset = ResPrototype("void   ext_param_iset( ext_param, int, double)")
    _iiset = ResPrototype("void   ext_param_iiset( ext_param, int, int, double)")
    _key_set = ResPrototype("void   ext_param_key_set( ext_param, char*, double)")
    _key_suffix_set = ResPrototype(
        "void   ext_param_key_suffix_set( ext_param, char*, char*, double)"
    )
    _iget = ResPrototype("double ext_param_iget( ext_param, int)")
    _iiget = ResPrototype("double ext_param_iiget( ext_param, int, int)")
    _key_get = ResPrototype("double ext_param_key_get( ext_param, char*)")
    _key_suffix_get = ResPrototype(
        "double ext_param_key_suffix_get( ext_param, char*, char*)"
    )
    _export = ResPrototype("void   ext_param_json_export( ext_param, char*)")
    _get_config = ResPrototype("void* ext_param_get_config(ext_param)")

    def __init__(self, config):
        c_ptr = self._alloc(config)
        super(ExtParam, self).__init__(c_ptr)

    def __contains__(self, key):
        return key in self.config

    def __len__(self):
        return len(self.config)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            # if the index is key suffix, assume they are both strings
            key, suffix = index
            if not isinstance(key, string_types) or not isinstance(
                suffix, string_types
            ):
                raise TypeError("Expected a pair of strings, got {}".format(index))
            self._check_key_suffix(key, suffix)
            return self._key_suffix_get(key, suffix)

        # index is just the key, it can be either a string or an int
        if isinstance(index, string_types):
            self._check_key_suffix(index)
            return self._key_get(index)

        index = self._roll_key_index(index)
        self._check_index(index)
        return self._iget(index)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            # if the index is key suffix, assume they are both strings
            key, suffix = index
            if not isinstance(key, string_types) or not isinstance(
                suffix, string_types
            ):
                raise TypeError("Expected a pair of strings, got {}".format(index))
            self._check_key_suffix(key, suffix)
            self._key_suffix_set(key, suffix, value)
            return

        # index is just the key, it can be either a string or an int
        if isinstance(index, string_types):
            self._check_key_suffix(index)
            self._key_set(index, value)
        else:
            index = self._roll_key_index(index)
            self._check_index(index)
            self._iset(index, value)

    def _roll_key_index(self, index):
        """ Support indexing from the end of the list of keys """
        return index if index >= 0 else index + len(self)

    def _check_index(self, kidx, sidx=None):
        """Raise if any of the following is true:
        - kidx is not a valid index for keys
        - the key referred to by kidx has no suffixes, but sidx is given
        - the key referred to by kidx has suffixes, but sidx is None
        - the key referred to by kidx has suffixes, and sidx is not a valid
          suffix index
        """
        if kidx < 0 or kidx >= len(self):
            raise IndexError(
                "Invalid key index {}. Valid range is [0, {})".format(kidx, len(self))
            )
        key, suffixes = self.config[kidx]
        if not suffixes:
            if sidx is None:
                return  # we are good
            raise IndexError(
                "Key {} has no suffixes, but suffix {} requested".format(key, sidx)
            )
        assert len(suffixes) > 0
        if sidx is None:
            raise IndexError(
                "Key {} has suffixes, a suffix index must be specified".format(key)
            )
        if sidx < 0 or sidx >= len(suffixes):
            raise IndexError(
                (
                    "Suffix index {} is out of range for key {}. Valid range is "
                    "[0, {})"
                ).format(sidx, key, len(suffixes))
            )

    def _check_key_suffix(self, key, suffix=None):
        """Raise if any of the following is true:
        - key is not present in config
        - key has no suffixes but a suffix is given
        - key has suffixes but suffix is None
        - key has suffixes but suffix is not among them
        """
        if not key in self:
            raise KeyError("No such key: {}".format(key))
        suffixes = self.config[key]
        if not suffixes:
            if suffix is None:
                return
            raise KeyError(
                "Key {} has no suffixes, but suffix {} requested".format(key, suffix)
            )
        assert len(suffixes) > 0
        if suffix is None:
            raise KeyError(
                "Key {} has suffixes, a suffix must be specified".format(key)
            )
        if suffix not in suffixes:
            raise KeyError(
                "Key {} has suffixes {}. Can't find the requested suffix {}".format(
                    key, suffixes, suffix
                )
            )

    @property
    def config(self):
        return ExtParamConfig.createCReference(self._get_config(), self)

    # This could in the future be specialized to take a numpy vector,
    # which could be vector-assigned in C.
    def set_vector(self, values):
        if len(values) != len(self):
            raise ValueError("Size mismatch")

        for index, value in enumerate(values):
            self[index] = value

    def free(self):
        self._free()

    def export(self, fname):
        self._export(fname)
