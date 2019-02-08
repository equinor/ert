#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'subst_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from os.path import isfile

from cwrap import BaseCClass
from res import ResPrototype

class SubstConfig(BaseCClass):
    TYPE_NAME = "subst_config"
    _alloc          = ResPrototype("void* subst_config_alloc(config_content)", bind=False)
    _free           = ResPrototype("void  subst_config_free(subst_config)")
    _get_subst_list = ResPrototype("subst_list_ref subst_config_get_subst_list( subst_config )")

    def __init__(self, config_content):
        c_ptr = self._alloc(config_content)

        if c_ptr is None:
            raise ValueError('Failed to construct RNGConfig instance')

        super(SubstConfig, self).__init__(c_ptr)

    def __getitem__(self, key):
        subst_list = self._get_subst_list( )
        return subst_list[key]

    def __iter__(self):
        subst_list = self._get_subst_list( )
        return iter(subst_list)

    @property
    def subst_list(self):
        return self._get_subst_list().setParent(self)

    def free(self):
        self._free()
