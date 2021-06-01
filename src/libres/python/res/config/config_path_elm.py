#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'config_path_elm.py' is part of ERT - Ensemble based Reservoir Tool.
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


class ConfigPathElm(BaseCClass):
    TYPE_NAME = "config_path_elm"

    _free = ResPrototype("void config_path_elm_free(config_path_elm)")
    _rel_path = ResPrototype("char* config_path_elm_get_relpath(config_path_elm)")
    _abs_path = ResPrototype("char* config_path_elm_get_abspath(config_path_elm)")

    def __init__(self):
        raise NotImplementedError("Not possible to instantiate ConfigPathElm directly.")

    def free(self):
        self._free()

    @property
    def rel_path(self):
        return self._rel_path()

    @property
    def abs_path(self):
        return self._abs_path()
