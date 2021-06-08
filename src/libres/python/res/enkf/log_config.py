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

from os.path import isfile, realpath

from cwrap import BaseCClass

from res.util.enums import MessageLevelEnum

from res import ResPrototype
from res.enkf import ConfigKeys


class LogConfig(BaseCClass):
    TYPE_NAME = "log_config"

    _alloc = ResPrototype("void* log_config_alloc(config_content)", bind=False)
    _alloc_load = ResPrototype("void* log_config_alloc_load(char*)", bind=False)
    _alloc_full = ResPrototype(
        "void* log_config_alloc_full(char*, message_level_enum)", bind=False
    )
    _free = ResPrototype("void log_config_free(log_config)")
    _log_file = ResPrototype("char* log_config_get_log_file(log_config)")
    _log_level = ResPrototype("message_level_enum log_config_get_log_level(log_config)")

    def __init__(self, user_config_file=None, config_content=None, config_dict=None):
        configs = sum(
            [
                1
                for x in [user_config_file, config_content, config_dict]
                if x is not None
            ]
        )

        if configs > 1:
            raise IOError(
                "Attempting to construct LogConfig with multiple config objects"
            )

        if configs == 0:
            raise IOError("Attempting to construct LogConfig with no config objects")

        c_ptr = None
        if user_config_file:
            if not isfile(user_config_file):
                raise IOError('No such configuration file "%s".' % user_config_file)
            c_ptr = self._alloc_load(user_config_file)

        if config_content:
            c_ptr = self._alloc(config_content)

        if config_dict is not None:

            if ConfigKeys.LOG_FILE in config_dict:
                log_file = realpath(config_dict[ConfigKeys.LOG_FILE])
            else:
                raise ValueError("No log file provided")

            if ConfigKeys.LOG_LEVEL in config_dict:
                message_level = config_dict[ConfigKeys.LOG_LEVEL]
            else:
                raise ValueError("No log level provided")

            c_ptr = self._alloc_full(log_file, message_level)

        if c_ptr is None:
            raise ValueError("Failed to construct LogConfig instance")
        super(LogConfig, self).__init__(c_ptr)

    def __repr__(self):
        return "LogConfig(log_file=%s, log_level=%r)" % (self.log_file, self.log_level)

    def free(self):
        self._free()

    @property
    def log_file(self):
        return self._log_file()

    @property
    def log_level(self):
        return self._log_level()

    def __eq__(self, other):
        if self.log_file != other.log_file:
            return False
        if self.log_level != other.log_level:
            return False
        return True
