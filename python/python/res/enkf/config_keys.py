#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  The file 'config_keys.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.enkf import EnkfPrototype
from enum import Enum

class ConfigKeys:

    _config_directory_key = EnkfPrototype("char* config_keys_get_config_directory_key()", bind=False)
    _queue_system_key     = EnkfPrototype("char* config_keys_get_queue_system_key()", bind=False)

    CONFIG_DIRECTORY = _config_directory_key()
    DEFINES          = "DEFINES"
    INTERNALS        = "INTERNALS"
    SIMULATION       = "SIMULATION"
    QUEUE_SYSTEM     = _queue_system_key()
