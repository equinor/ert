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
    _run_template_key     = EnkfPrototype("char* config_keys_get_run_template_key()", bind=False)
    _gen_kw_key           = EnkfPrototype("char* config_keys_get_gen_kw_key()", bind=False)
    _history_source_key   = EnkfPrototype("char* config_keys_get_history_source_key()", bind=False)
    _queue_option_key     = EnkfPrototype("char* config_keys_get_queue_option_key()", bind=False)
    _install_job_key      = EnkfPrototype("char* config_keys_get_install_job_key()", bind=False)
    _plot_settings_key    = EnkfPrototype("char* config_keys_get_plot_setting_key()", bind=False)


    CONFIG_DIRECTORY = _config_directory_key()
    DEFINES          = "DEFINES"
    INTERNALS        = "INTERNALS"
    SIMULATION       = "SIMULATION"
    LOGGING          = "LOGGING"
    SEED             = "SEED"
    QUEUE_SYSTEM     = _queue_system_key()
    RUN_TEMPLATE     = _run_template_key()
    TEMPLATE         = "TEMPLATE"
    EXPORT           = "EXPORT"
    GEN_KW           = _gen_kw_key()
    NAME             = "NAME"
    OUT_FILE         = "OUT_FILE"
    PARAMETER_FILE   = "PARAMETER_FILE"
    PATH             = "PATH"
    QUEUE_OPTION     = _queue_option_key()
    DRIVER_NAME      = "DRIVER_NAME"
    OPTION           = "OPTION"
    VALUE            = "VALUE"
    INSTALL_JOB      = _install_job_key()
    PLOT_SETTINGS    = _plot_settings_key()
