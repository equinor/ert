#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'analysis_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from os.path import realpath

from cwrap import BaseCClass

from ecl.util.util import StringList

from res import ResPrototype
from res.enkf import ConfigKeys
from res.enkf import AnalysisIterConfig
from res.analysis import AnalysisModule


class AnalysisConfig(BaseCClass):
    TYPE_NAME = "analysis_config"

    _alloc = ResPrototype("void* analysis_config_alloc(config_content)", bind=False)
    _alloc_load = ResPrototype("void* analysis_config_alloc_load(char*)", bind=False)
    _alloc_full = ResPrototype(
        "void* analysis_config_alloc_full(double, bool, "
        "bool, int, char*, double, bool, bool, "
        "bool, double, int, int)",
        bind=False,
    )

    _add_module_copy = ResPrototype(
        "void analysis_config_add_module_copy( analysis_config, " "char* , char* )"
    )
    _load_external_module = ResPrototype(
        "bool analysis_config_load_external_module( analysis_config, " "char* , char* )"
    )

    _free = ResPrototype("void analysis_config_free( analysis_config )")
    _get_rerun = ResPrototype("int analysis_config_get_rerun( analysis_config )")
    _set_rerun = ResPrototype("void analysis_config_set_rerun( analysis_config, bool)")
    _get_rerun_start = ResPrototype(
        "int analysis_config_get_rerun_start( analysis_config )"
    )
    _set_rerun_start = ResPrototype(
        "void analysis_config_set_rerun_start( analysis_config, int)"
    )
    _get_log_path = ResPrototype("char* analysis_config_get_log_path( analysis_config)")
    _set_log_path = ResPrototype(
        "void analysis_config_set_log_path( analysis_config, char*)"
    )
    _get_merge_observations = ResPrototype(
        "bool analysis_config_get_merge_observations(analysis_config)"
    )
    _set_merge_observations = ResPrototype(
        "void analysis_config_set_merge_observations(analysis_config, bool)"
    )
    _get_iter_config = ResPrototype(
        "analysis_iter_config_ref analysis_config_get_iter_config(analysis_config)"
    )
    _have_enough_realisations = ResPrototype(
        "bool analysis_config_have_enough_realisations(analysis_config, int, int)"
    )
    _get_max_runtime = ResPrototype(
        "int analysis_config_get_max_runtime(analysis_config)"
    )
    _set_max_runtime = ResPrototype(
        "void analysis_config_set_max_runtime(analysis_config, int)"
    )
    _get_stop_long_running = ResPrototype(
        "bool analysis_config_get_stop_long_running(analysis_config)"
    )
    _set_stop_long_running = ResPrototype(
        "void analysis_config_set_stop_long_running(analysis_config, bool)"
    )
    _get_active_module_name = ResPrototype(
        "char* analysis_config_get_active_module_name(analysis_config)"
    )
    _get_module_list = ResPrototype(
        "stringlist_obj analysis_config_alloc_module_names(analysis_config)"
    )
    _get_module = ResPrototype(
        "analysis_module_ref analysis_config_get_module(analysis_config, char*)"
    )
    _select_module = ResPrototype(
        "bool analysis_config_select_module(analysis_config, char*)"
    )
    _has_module = ResPrototype(
        "bool analysis_config_has_module(analysis_config, char*)"
    )
    _get_alpha = ResPrototype("double analysis_config_get_alpha(analysis_config)")
    _set_alpha = ResPrototype("void analysis_config_set_alpha(analysis_config, double)")
    _get_std_cutoff = ResPrototype(
        "double analysis_config_get_std_cutoff(analysis_config)"
    )
    _set_std_cutoff = ResPrototype(
        "void analysis_config_set_std_cutoff(analysis_config, double)"
    )
    _set_global_std_scaling = ResPrototype(
        "void analysis_config_set_global_std_scaling(analysis_config, double)"
    )
    _get_global_std_scaling = ResPrototype(
        "double analysis_config_get_global_std_scaling(analysis_config)"
    )
    _get_min_realizations = ResPrototype(
        "int analysis_config_get_min_realisations(analysis_config)"
    )

    def __init__(self, user_config_file=None, config_content=None, config_dict=None):
        configs = sum(
            [
                1
                for x in [user_config_file, config_content, config_dict]
                if x is not None
            ]
        )

        if configs > 1:
            raise ValueError(
                "Attempting to create AnalysisConfig object with multiple config objects"
            )

        if configs == 0:
            raise ValueError(
                "Error trying to create AnalysisConfig without any configuration"
            )

        c_ptr = None

        if user_config_file is not None:
            if not isfile(user_config_file):
                raise IOError('No such configuration file "%s".' % user_config_file)

            c_ptr = self._alloc_load(user_config_file)
            if c_ptr:
                super(AnalysisConfig, self).__init__(c_ptr)
            else:
                raise ValueError(
                    "Failed to construct AnalysisConfig instance from config file %s."
                    % user_config_file
                )

        if config_content is not None:
            c_ptr = self._alloc(config_content)
            if c_ptr:
                super(AnalysisConfig, self).__init__(c_ptr)
            else:
                raise ValueError("Failed to construct AnalysisConfig instance.")

        if config_dict is not None:
            c_ptr = self._alloc_full(
                config_dict.get(ConfigKeys.ALPHA_KEY, 3.0),
                config_dict.get(ConfigKeys.MERGE_OBSERVATIONS, False),
                config_dict.get(ConfigKeys.RERUN_KEY, False),
                config_dict.get(ConfigKeys.RERUN_START_KEY, 0),
                realpath(config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log")),
                config_dict.get(ConfigKeys.STD_CUTOFF_KEY, 1e-6),
                config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False),
                config_dict.get(ConfigKeys.SINGLE_NODE_UPDATE, False),
                config_dict.get(ConfigKeys.STD_CORRELATED_OBS, False),
                config_dict.get(ConfigKeys.GLOBAL_STD_SCALING, 1.0),
                config_dict.get(ConfigKeys.MAX_RUNTIME, 0),
                config_dict.get(ConfigKeys.MIN_REALIZATIONS, 0),
            )
            if c_ptr:
                super(AnalysisConfig, self).__init__(c_ptr)

                # external modules
                ext_modules_list = config_dict.get(ConfigKeys.ANALYSIS_LOAD, [])
                for ext_module in ext_modules_list:
                    self._load_external_module(
                        ext_module[ConfigKeys.LIB_NAME],
                        ext_module[ConfigKeys.USER_NAME],
                    )

                # copy modules
                analysis_copy_list = config_dict.get(ConfigKeys.ANALYSIS_COPY, [])
                for analysis_copy in analysis_copy_list:
                    self._add_module_copy(
                        analysis_copy[ConfigKeys.SRC_NAME],
                        analysis_copy[ConfigKeys.DST_NAME],
                    )

                # set var list
                set_var_list = config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, [])
                for set_var in set_var_list:
                    module = self._get_module(set_var[ConfigKeys.MODULE_NAME])
                    module._set_var(
                        set_var[ConfigKeys.VAR_NAME], str(set_var[ConfigKeys.VALUE])
                    )

                if ConfigKeys.ANALYSIS_SELECT in config_dict:
                    self._select_module(config_dict[ConfigKeys.ANALYSIS_SELECT])

            else:
                raise ValueError("Failed to construct AnalysisConfig from dict.")

    def get_rerun(self):
        return self._get_rerun()

    def set_rerun(self, rerun):
        self._set_rerun(rerun)

    def get_rerun_start(self):
        return self._get_rerun_start()

    def set_rerun_start(self, index):
        self._set_rerun_start(index)

    def get_log_path(self):
        return self._get_log_path()

    def set_log_path(self, path):
        self._set_log_path(path)

    def getEnkfAlpha(self):
        """ :rtype: float """
        return self._get_alpha()

    def setEnkfAlpha(self, alpha):
        self._set_alpha(alpha)

    def getStdCutoff(self):
        """ :rtype: float """
        return self._get_std_cutoff()

    def setStdCutoff(self, std_cutoff):
        self._set_std_cutoff(std_cutoff)

    def get_merge_observations(self):
        return self._get_merge_observations()

    def set_merge_observations(self, merge_observations):
        return self._set_merge_observations(merge_observations)

    def getAnalysisIterConfig(self):
        """ @rtype: AnalysisIterConfig """
        return self._get_iter_config().setParent(self)

    def get_stop_long_running(self):
        """ @rtype: bool """
        return self._get_stop_long_running()

    def set_stop_long_running(self, stop_long_running):
        self._set_stop_long_running(stop_long_running)

    def get_max_runtime(self):
        """ @rtype: int """
        return self._get_max_runtime()

    def set_max_runtime(self, max_runtime):
        self._set_max_runtime(max_runtime)

    def free(self):
        self._free()

    def activeModuleName(self):
        """ :rtype: str """
        return self._get_active_module_name()

    def getModuleList(self):
        """ :rtype: StringList """
        return self._get_module_list()

    def getModule(self, module_name):
        """ @rtype: AnalysisModule """
        return self._get_module(module_name)

    def hasModule(self, module_name):
        """ @rtype: bool """
        return self._has_module(module_name)

    def selectModule(self, module_name):
        """ @rtype: bool """
        return self._select_module(module_name)

    def getActiveModule(self):
        """ :rtype: AnalysisModule """
        return self.getModule(self.activeModuleName())

    def setGlobalStdScaling(self, std_scaling):
        self._set_global_std_scaling(std_scaling)

    def getGlobalStdScaling(self):
        return self._get_global_std_scaling()

    @property
    def minimum_required_realizations(self):
        return self._get_min_realizations()

    def haveEnoughRealisations(self, realizations, ensemble_size):
        return realizations >= min(self.minimum_required_realizations, ensemble_size)

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if realpath(self.get_log_path()) != realpath(other.get_log_path()):
            return False

        if self.get_max_runtime() != other.get_max_runtime():
            return False

        if self.getGlobalStdScaling() != other.getGlobalStdScaling():
            return False

        if self.get_stop_long_running() != other.get_stop_long_running():
            return False

        if self.getStdCutoff() != other.getStdCutoff():
            return False

        if self.getEnkfAlpha() != other.getEnkfAlpha():
            return False

        if self.get_merge_observations() != other.get_merge_observations():
            return False

        if self.get_rerun() != other.get_rerun():
            return False

        if self.get_rerun_start() != other.get_rerun_start():
            return False

        if set(self.getModuleList()) != set(other.getModuleList()):
            return False

        if self.activeModuleName() != other.activeModuleName():
            return False

        if self.getAnalysisIterConfig() != other.getAnalysisIterConfig():
            return False

        # compare each module
        for a in list(self.getModuleList()):
            if self.getModule(a) != other.getModule(a):
                return False

        return True
