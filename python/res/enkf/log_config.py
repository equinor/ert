#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'log_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.util.enums import MessageLevelEnum

from res import ResPrototype

class LogConfig(BaseCClass):
    TYPE_NAME = "log_config"

    _alloc      = ResPrototype("void* log_config_alloc(config_content)", bind=False)
    _alloc_load = ResPrototype("void* log_config_alloc_load(char*)", bind=False)
    _free       = ResPrototype("void log_config_free(log_config)")
    _log_file   = ResPrototype("char* log_config_get_log_file(log_config)")
    _log_level  = ResPrototype("message_level_enum log_config_get_log_level(log_config)")

    def __init__(self, user_config_file = None, config_content = None):
        if config_content is None:
            if user_config_file is not None and not isfile(user_config_file):
                raise IOError('No such configuration file "%s".' % user_config_file)

            c_ptr = self._alloc_load(user_config_file)
            if c_ptr:
                super(LogConfig, self).__init__(c_ptr)
            else:
                raise ValueError('Failed to construct LogConfig instance from config file %s.' % user_config_file)
        else:
            c_ptr = self._alloc(config_content)
            if c_ptr is None:
                raise ValueError('Failed to construct LogConfig instance')
            super(LogConfig, self).__init__(c_ptr)

    def __repr__(self):
        return "LogConfig(log_file=%s, log_level=%r)" % (
                self.log_file, self.log_level)

    def free(self):
        self._free()

    @property
    def log_file(self):
        return self._log_file()

    @property
    def log_level(self):
        return self._log_level()
