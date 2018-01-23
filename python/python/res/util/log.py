#  Copyright (C) 2012  Statoil ASA, Norway. 
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

from __future__ import print_function
from cwrap import BaseCClass
from res.util import ResUtilPrototype
from res.util.enums import MessageLevelEnum


class Log(BaseCClass):
    _get_filename = ResUtilPrototype("char* log_get_filename(log)")
    _get_level = ResUtilPrototype("message_level_enum log_get_level(log)")
    _set_level = ResUtilPrototype("void log_set_level(log, message_level_enum)")

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def get_filename(self):
        return self._get_filename()
        # return "ert_config.log"

    def get_level(self):
        return self._get_level()

    def set_level(self, level):
        self._set_level(self, level)
