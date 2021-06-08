#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'log.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.util.enums import MessageLevelEnum


class Log(BaseCClass):
    _open_log = ResPrototype(
        "void* log_open_file(char*, message_level_enum)", bind=False
    )
    _get_filename = ResPrototype("char* log_get_filename(log)")
    _set_level = ResPrototype("void log_set_level(log, message_level_enum)")

    def __init__(self, log_filename, log_level):
        c_ptr = self._open_log(log_filename, log_level)
        if c_ptr:
            super(Log, self).__init__(c_ptr)
        else:
            raise IOError("Failed to open log handle at:%s" % log_filename)

    def get_filename(self):
        return self._get_filename()

    def set_level(self, level):
        self._set_level(self, level)
