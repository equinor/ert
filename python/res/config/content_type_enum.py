#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'content_type_enum.py' is part of ERT - Ensemble based Reservoir Tool.
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

import ctypes
from cwrap import BaseCEnum
from ecl import EclPrototype
from res import ResPrototype


class ContentTypeEnum(BaseCEnum):
    TYPE_NAME = "config_content_type_enum"
    CONFIG_STRING = None
    CONFIG_INT = None
    CONFIG_FLOAT = None
    CONFIG_PATH = None
    CONFIG_EXISTING_PATH = None
    CONFIG_BOOL = None
    CONFIG_CONFIG = None
    CONFIG_BYTESIZE = None
    CONFIG_EXECUTABLE = None
    CONFIG_ISODATE = None
    CONFIG_INVALID = None
    CONFIG_RUNTIME_FILE = None
    CONFIG_RUNTIME_INT = None

    _valid_string = ResPrototype(
        "bool config_schema_item_valid_string(config_content_type_enum ,  char*, bool)"
    )
    _sscanf_bool = EclPrototype("bool util_sscanf_bool( char* , bool*)", bind=False)

    def valid_string(self, string, runtime=False):
        return self._valid_string(string, runtime)

    def convert_string(self, string):
        if not self.valid_string(string, runtime=True):
            raise ValueError("Can not convert %s to %s" % (string, self))

        if self == ContentTypeEnum.CONFIG_INT:
            return int(string)

        if self == ContentTypeEnum.CONFIG_FLOAT:
            return float(string)

        if self == ContentTypeEnum.CONFIG_BOOL:
            bool_value = ctypes.c_bool()
            ContentTypeEnum._sscanf_bool(string, ctypes.byref(bool_value))
            return bool_value.value

        return string


ContentTypeEnum.addEnum("CONFIG_STRING", 1)
ContentTypeEnum.addEnum("CONFIG_INT", 2)
ContentTypeEnum.addEnum("CONFIG_FLOAT", 4)
ContentTypeEnum.addEnum("CONFIG_PATH", 8)
ContentTypeEnum.addEnum("CONFIG_EXISTING_PATH", 16)
ContentTypeEnum.addEnum("CONFIG_BOOL", 32)
ContentTypeEnum.addEnum("CONFIG_CONFIG", 64)
ContentTypeEnum.addEnum("CONFIG_BYTESIZE", 128)
ContentTypeEnum.addEnum("CONFIG_EXECUTABLE", 256)
ContentTypeEnum.addEnum("CONFIG_ISODATE", 512)
ContentTypeEnum.addEnum("CONFIG_INVALID", 1024)
ContentTypeEnum.addEnum("CONFIG_RUNTIME_INT", 2048)
ContentTypeEnum.addEnum("CONFIG_RUNTIME_FILE", 4096)
