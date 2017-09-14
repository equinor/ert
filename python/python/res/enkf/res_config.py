#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  The file 'res_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

import os
from os.path import isfile

from cwrap import BaseCClass

from ecl.util import StringList

from res.config import (ConfigParser, ConfigContent, ConfigSettings,
                        UnrecognizedEnum)
from res.enkf import EnkfPrototype
from res.enkf import (SiteConfig, AnalysisConfig, SubstConfig, ModelConfig, EclConfig,
                      EnsembleConfig, RNGConfig)

class ResConfig(BaseCClass):

    TYPE_NAME = "res_config"

    _alloc_load = EnkfPrototype("void* res_config_alloc_load(char*)", bind=False)
    _alloc      = EnkfPrototype("void* res_config_alloc(config_content)", bind=False)
    _free       = EnkfPrototype("void res_config_free(res_config)")

    _user_config_file  = EnkfPrototype("char* res_config_get_user_config_file(res_config)")

    _config_path       = EnkfPrototype("char* res_config_get_config_directory(res_config)")
    _site_config       = EnkfPrototype("site_config_ref res_config_get_site_config(res_config)")
    _analysis_config   = EnkfPrototype("analysis_config_ref res_config_get_analysis_config(res_config)")
    _subst_config      = EnkfPrototype("subst_config_ref res_config_get_subst_config(res_config)")
    _model_config      = EnkfPrototype("model_config_ref res_config_get_model_config(res_config)")
    _ecl_config        = EnkfPrototype("ecl_config_ref res_config_get_ecl_config(res_config)")
    _ensemble_config   = EnkfPrototype("ens_config_ref res_config_get_ensemble_config(res_config)")
    _plot_config       = EnkfPrototype("plot_settings_ref res_config_get_plot_config(res_config)")
    _hook_manager      = EnkfPrototype("hook_manager_ref res_config_get_hook_manager(res_config)")
    _ert_workflow_list = EnkfPrototype("ert_workflow_list_ref res_config_get_workflow_list(res_config)")
    _rng_config        = EnkfPrototype("rng_config_ref res_config_get_rng_config(res_config)")
    _ert_templates     = EnkfPrototype("ert_templates_ref res_config_get_templates(res_config)")
    _log_config        = EnkfPrototype("log_config_ref res_config_get_log_config(res_config)")
    _add_config_items  = EnkfPrototype("void res_config_add_config_items(config_parser)")
    _init_parser       = EnkfPrototype("void res_config_init_config_parser(config_parser)", bind=False)

    def __init__(self, user_config_file=None, config=None, throw_on_error=True):
        self._errors, self._failed_keys = None, None
        self._assert_input(user_config_file, config, throw_on_error)

        if config is not None:
            config_content = self._build_config_content(config)

            c_ptr = None
            if not self.errors or not throw_on_error:
                c_ptr = self._alloc(config_content)
        else:
            c_ptr = self._alloc_load(user_config_file)

        if c_ptr:
            super(ResConfig, self).__init__(c_ptr)
        else:
            raise ValueError(
                    'Failed to construct ResConfig instance from %r.'
                    % user_config_file if user_config_file else config
                    )


    def _assert_input(self, user_config_file, config, throw_on_error):
        if config and not isinstance(config, dict):
            raise ValueError("Expected config to be a dictionary, was %r"
                             % type(config))

        if user_config_file and not isinstance(user_config_file, str):
            raise ValueError("Expected user_config_file to be a string.")

        if user_config_file is not None and config is not None:
            raise ValueError("Expected either user_config_file " +
                             "or config to be provided, got both!")

        if user_config_file is not None and not isfile(user_config_file):
            raise IOError('No such configuration file "%s".' % user_config_file)

        if user_config_file is not None and not throw_on_error:
            raise NotImplementedError("Disabling exceptions on errors is not "
                                      "available when loading from file.")


    def _build_config_content(self, config):
        self._failed_keys = {}
        config_parser  = ConfigParser()
        ResConfig.init_config_parser(config_parser)

        config_content = ConfigContent(None)
        config_content.setParser(config_parser)

        config["WORKING_DIRECTORY"] = os.path.realpath(config["WORKING_DIRECTORY"])
        path_elm = config_content.create_path_elm(config["WORKING_DIRECTORY"])

        for key in config.keys():
            value = str(config[key]) # TODO: Support lists of arguments
            ok = config_parser.add_key_value(config_content,
                                        key,
                                        StringList([key, value]),
                                        path_elm=path_elm,
                                        )

            if not ok:
                self._failed_keys[key] = config[key]

        config_parser.validate(config_content)
        self._errors = list(config_content.getErrors())

        return config_content

    def free(self):
        self._free()

    @classmethod
    def init_config_parser(cls, config_parser):
        cls._init_parser(config_parser)

    @property
    def errors(self):
        return self._errors

    @property
    def failed_keys(self):
        return self._failed_keys

    @property
    def user_config_file(self):
        return self._user_config_file()

    @property
    def site_config_file(self):
        return self.site_config.config_file

    @property
    def site_config(self):
        return self._site_config()

    @property
    def analysis_config(self):
        return self._analysis_config()

    @property
    def config_path(self):
        return self._config_path( )

    @property
    def subst_config(self):
        return self._subst_config( )

    @property
    def model_config(self):
        return self._model_config()

    @property
    def ecl_config(self):
        return self._ecl_config()

    @property
    def ensemble_config(self):
        return self._ensemble_config()

    @property
    def plot_config(self):
        return self._plot_config()

    @property
    def hook_manager(self):
        return self._hook_manager()

    @property
    def ert_workflow_list(self):
        return self._ert_workflow_list()

    @property
    def rng_config(self):
        return self._rng_config()

    @property
    def ert_templates(self):
        return self._ert_templates()

    @property
    def log_config(self):
        return self._log_config()
