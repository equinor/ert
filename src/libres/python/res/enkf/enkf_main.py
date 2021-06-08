#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ecl_kw.py' is part of ERT - Ensemble based Reservoir Tool.
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
import sys
import ctypes, warnings
from os.path import isfile

from cwrap import BaseCClass
from res import ResPrototype
from res.enkf import (
    AnalysisConfig,
    EclConfig,
    LocalConfig,
    ModelConfig,
    EnsembleConfig,
    SiteConfig,
    ResConfig,
    QueueConfig,
)
from res.enkf import EnkfObs, EnKFState, EnkfSimulationRunner, EnkfFsManager
from res.enkf import ErtWorkflowList, HookManager, HookWorkflow, ESUpdate
from res.enkf.enums import EnkfInitModeEnum
from res.enkf.key_manager import KeyManager
from res.util import Log
from res.util.substitution_list import SubstitutionList
from ecl.util.util import rng


class EnKFMain(BaseCClass):

    TYPE_NAME = "enkf_main"

    @classmethod
    def createPythonObject(cls, c_pointer):
        if c_pointer is not None:
            real_enkf_main = _RealEnKFMain.createPythonObject(c_pointer)
            new_obj = cls.__new__(cls)
            EnKFMain._init_from_real_enkf_main(new_obj, real_enkf_main)
            EnKFMain._monkey_patch_methods(new_obj, real_enkf_main)
            return new_obj
        else:
            return None

    @classmethod
    def createCReference(cls, c_pointer, parent=None):
        if c_pointer is not None:
            real_enkf_main = _RealEnKFMain.createCReference(c_pointer, parent)
            new_obj = cls.__new__(cls)
            EnKFMain._init_from_real_enkf_main(new_obj, real_enkf_main)
            EnKFMain._monkey_patch_methods(new_obj, real_enkf_main)
            return new_obj
        else:
            return None

    def __init__(self, config, strict=True, verbose=False):
        """Initializes an instance of EnkfMain.

        Note: @config is a ResConfig instance holding the configuration.
        """

        real_enkf_main = _RealEnKFMain(config, strict, verbose)
        assert isinstance(real_enkf_main, BaseCClass)
        self._init_from_real_enkf_main(real_enkf_main)
        self._monkey_patch_methods(real_enkf_main)

    def _init_from_real_enkf_main(self, real_enkf_main):
        super(EnKFMain, self).__init__(
            real_enkf_main.from_param(real_enkf_main).value,
            parent=real_enkf_main,
            is_reference=True,
        )

        self.__simulation_runner = EnkfSimulationRunner(self)
        self.__fs_manager = EnkfFsManager(self)
        self.__es_update = ESUpdate(self)

    def _real_enkf_main(self):
        return self.parent()

    def getESUpdate(self):
        """@rtype: ESUpdate"""
        return self.__es_update

    def getEnkfSimulationRunner(self):
        """@rtype: EnkfSimulationRunner"""
        return self.__simulation_runner

    def getEnkfFsManager(self):
        """@rtype: EnkfFsManager"""
        return self.__fs_manager

    def umount(self):
        if self.__fs_manager is not None:
            self.__fs_manager.umount()

    # --- Overridden methods --------------------

    def _monkey_patch_methods(self, real_enkf_main):
        # As a general rule, EnKFMain methods should be implemented on
        # _RealEnKFMain because the other references (such as __es_update)
        # may need to use them.
        # The public methods should be also exposed in this class, forwarding
        # the call to the real method on the real_enkf_main object. That's done
        # via monkey patching, so we don't need to manually keep the classes
        # synchronized
        from inspect import getmembers, ismethod
        from functools import partial

        methods = getmembers(self._real_enkf_main(), predicate=ismethod)
        dont_patch = [name for name, _ in getmembers(BaseCClass)]
        for name, method in methods:
            if name.startswith("_") or name in dont_patch:
                continue  # skip private methods
            setattr(self, name, method)

    def __repr__(self):
        repr = self._real_enkf_main().__repr__()
        assert repr.startswith("_RealEnKFMain")
        return repr[5:]


class _RealEnKFMain(BaseCClass):
    """Access to the C EnKFMain interface.

    The python interface of EnKFMain is split between 4 classes, ie
    - EnKFMain: main entry point, defined further down
    - EnkfSimulationRunner, EnkfFsManager and ESUpdate: access specific
      functionalities
    EnKFMain owns an instance of each of the last 3 classes. Also, all
    of these classes need to access the same underlying C object.
    So, in order to avoid circular dependencies, we make _RealEnKF main
    the only "owner" of the C object, and all the classes that need to
    access it set _RealEnKFMain as parent.

    The situation can be summarized as follows (show only EnkfFSManager,
    classes EnkfSimulationRunner and ESUpdate are treated analogously)
     ------------------------------------
    |   real EnKFMain object in memory   |
     ------------------------------------
          ^                 ^          ^
          |                 |          |
       (c_ptr)           (c_ptr)       |
          |                 |          |
    _RealEnKFMain           |          |
       ^   ^                |          |
       |   ^--(parent)-- EnKFMain      |
       |                    |          |
       |                  (owns)       |
    (parent)                |       (c_ptr)
       |                    v          |
        ------------ EnkfFSManager ----

    """

    _alloc = ResPrototype("void* enkf_main_alloc(res_config, bool, bool)", bind=False)

    _free = ResPrototype("void enkf_main_free(enkf_main)")
    _get_queue_config = ResPrototype(
        "queue_config_ref enkf_main_get_queue_config(enkf_main)"
    )
    _get_ensemble_size = ResPrototype("int enkf_main_get_ensemble_size( enkf_main )")
    _get_ens_config = ResPrototype(
        "ens_config_ref enkf_main_get_ensemble_config( enkf_main )"
    )
    _get_model_config = ResPrototype(
        "model_config_ref enkf_main_get_model_config( enkf_main )"
    )
    _get_local_config = ResPrototype(
        "local_config_ref enkf_main_get_local_config( enkf_main )"
    )
    _get_analysis_config = ResPrototype(
        "analysis_config_ref enkf_main_get_analysis_config( enkf_main)"
    )
    _get_site_config = ResPrototype(
        "site_config_ref enkf_main_get_site_config( enkf_main)"
    )
    _get_ecl_config = ResPrototype(
        "ecl_config_ref enkf_main_get_ecl_config( enkf_main)"
    )
    _get_schedule_prediction_file = ResPrototype(
        "char* enkf_main_get_schedule_prediction_file( enkf_main )"
    )
    _get_data_kw = ResPrototype("subst_list_ref enkf_main_get_data_kw(enkf_main)")
    _clear_data_kw = ResPrototype("void enkf_main_clear_data_kw(enkf_main)")
    _add_data_kw = ResPrototype("void enkf_main_add_data_kw(enkf_main, char*, char*)")
    _resize_ensemble = ResPrototype("void enkf_main_resize_ensemble(enkf_main, int)")
    _get_obs = ResPrototype("enkf_obs_ref enkf_main_get_obs(enkf_main)")
    _load_obs = ResPrototype("bool enkf_main_load_obs(enkf_main, char* , bool)")
    _get_templates = ResPrototype(
        "ert_templates_ref enkf_main_get_templates(enkf_main)"
    )
    _get_site_config_file = ResPrototype(
        "char* enkf_main_get_site_config_file(enkf_main)"
    )
    _get_history_length = ResPrototype("int enkf_main_get_history_length(enkf_main)")
    _get_observations = ResPrototype(
        "void enkf_main_get_observations(enkf_main, char*, int, long*, double*, double*)"
    )
    _get_observation_count = ResPrototype(
        "int enkf_main_get_observation_count(enkf_main, char*)"
    )
    _have_observations = ResPrototype("bool enkf_main_have_obs(enkf_main)")
    _iget_state = ResPrototype("enkf_state_ref enkf_main_iget_state(enkf_main, int)")
    _get_workflow_list = ResPrototype(
        "ert_workflow_list_ref enkf_main_get_workflow_list(enkf_main)"
    )
    _get_hook_manager = ResPrototype(
        "hook_manager_ref enkf_main_get_hook_manager(enkf_main)"
    )
    _get_user_config_file = ResPrototype(
        "char* enkf_main_get_user_config_file(enkf_main)"
    )
    _get_mount_point = ResPrototype("char* enkf_main_get_mount_root( enkf_main )")
    _export_field = ResPrototype(
        "bool enkf_main_export_field(enkf_main, char*, char*, bool_vector, enkf_field_file_format_enum, int)"
    )
    _export_field_with_fs = ResPrototype(
        "bool enkf_main_export_field_with_fs(enkf_main, char*, char*, bool_vector, enkf_field_file_format_enum, int, enkf_fs_manager)"
    )
    _load_from_forward_model = ResPrototype(
        "int enkf_main_load_from_forward_model_from_gui(enkf_main, int, bool_vector, enkf_fs)"
    )
    _load_from_run_context = ResPrototype(
        "int enkf_main_load_from_run_context_from_gui(enkf_main, ert_run_context, enkf_fs)"
    )
    _create_run_path = ResPrototype(
        "void enkf_main_create_run_path(enkf_main , ert_run_context)"
    )
    _submit_simulation = ResPrototype(
        "void enkf_main_isubmit_job(enkf_main , run_arg, job_queue)"
    )
    _alloc_run_context_ENSEMBLE_EXPERIMENT = ResPrototype(
        "ert_run_context_obj enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT( enkf_main , enkf_fs , bool_vector , int)"
    )
    _get_runpath_list = ResPrototype(
        "runpath_list_ref enkf_main_get_runpath_list(enkf_main)"
    )
    _create_runpath_list = ResPrototype(
        "runpath_list_obj enkf_main_alloc_runpath_list(enkf_main)"
    )
    _add_node = ResPrototype("void enkf_main_add_node(enkf_main, enkf_config_node)")
    _get_res_config = ResPrototype("res_config_ref enkf_main_get_res_config(enkf_main)")
    _init_run = ResPrototype("void enkf_main_init_run(enkf_main, ert_run_context)")
    _get_shared_rng = ResPrototype("rng_ref enkf_main_get_shared_rng(enkf_main)")

    def __init__(self, config, strict=True, verbose=False):
        """Please don't use this class directly. See EnKFMain instead"""

        res_config = self._init_res_config(config)
        if res_config is None:
            raise TypeError(
                "Failed to construct EnKFMain instance due to invalid res_config."
            )

        c_ptr = self._alloc(res_config, strict, verbose)
        if c_ptr:
            super(_RealEnKFMain, self).__init__(c_ptr)
        else:
            raise ValueError(
                "Failed to construct EnKFMain instance from config %s." % res_config
            )

        self.__key_manager = KeyManager(self)

    def _init_res_config(self, config):
        if isinstance(config, ResConfig):
            return config

        # The res_config argument can be None; the only reason to
        # allow that possibility is to be able to test that the
        # site-config loads correctly.
        if config is None:
            config = ResConfig(None)
            config.convertToCReference(self)
            return config

        raise TypeError("Expected ResConfig, received: %r" % config)

    def get_queue_config(self):
        return self._get_queue_config()

    def getRealisation(self, iens):
        """@rtype: EnKFState"""
        if 0 <= iens < self.getEnsembleSize():
            return self._iget_state(iens).setParent(self)
        else:
            raise IndexError(
                "iens value:%d invalid Valid range: [0,%d)"
                % (iens, self.getEnsembleSize())
            )

    def free(self):
        self._free()

    def __repr__(self):
        ens = self.getEnsembleSize()
        cfg = self.getUserConfigFile()
        cnt = "ensemble_size = %d, config_file = %s" % (ens, cfg)
        return self._create_repr(cnt)

    def getEnsembleSize(self):
        """@rtype: int"""
        return self._get_ensemble_size()

    def resizeEnsemble(self, value):
        self._resize_ensemble(value)

    def ensembleConfig(self):
        """@rtype: EnsembleConfig"""
        return self._get_ens_config().setParent(self)

    def analysisConfig(self):
        """@rtype: AnalysisConfig"""
        return self._get_analysis_config().setParent(self)

    def getModelConfig(self):
        """@rtype: ModelConfig"""
        return self._get_model_config().setParent(self)

    def getLocalConfig(self):
        """@rtype: LocalConfig"""
        config = self._get_local_config().setParent(self)
        config.initAttributes(
            self.ensembleConfig(), self.getObservations(), self.eclConfig().getGrid()
        )
        return config

    def siteConfig(self):
        """@rtype: SiteConfig"""
        return self._get_site_config().setParent(self)

    def resConfig(self):
        return self._get_res_config().setParent(self)

    def eclConfig(self):
        """@rtype: EclConfig"""
        return self._get_ecl_config().setParent(self)

    def get_schedule_prediction_file(self):
        schedule_prediction_file = self._get_schedule_prediction_file()
        return schedule_prediction_file

    def getDataKW(self):
        """@rtype: SubstitutionList"""
        return self._get_data_kw()

    def clearDataKW(self):
        self._clear_data_kw()

    def addDataKW(self, key, value):
        self._add_data_kw(key, value)

    def getMountPoint(self):
        return self._get_mount_point()

    def getObservations(self):
        """@rtype: EnkfObs"""
        return self._get_obs().setParent(self)

    def have_observations(self):
        return self._have_observations()

    def loadObservations(self, obs_config_file, clear=True):
        return self._load_obs(obs_config_file, clear)

    def get_templates(self):
        return self._get_templates().setParent(self)

    def get_site_config_file(self):
        return self._get_site_config_file()

    def getUserConfigFile(self):
        """@rtype: str"""
        config_file = self._get_user_config_file()
        return config_file

    def getHistoryLength(self):
        return self._get_history_length()

    def getMemberRunningState(self, ensemble_member):
        """@rtype: EnKFState"""
        return self._iget_state(ensemble_member).setParent(self)

    def get_observations(self, user_key, obs_count, obs_x, obs_y, obs_std):
        return self._get_observations(user_key, obs_count, obs_x, obs_y, obs_std)

    def get_observation_count(self, user_key):
        return self._get_observation_count(user_key)

    def getKeyManager(self):
        """:rtype: KeyManager"""
        return self.__key_manager

    def getWorkflowList(self):
        """@rtype: ErtWorkflowList"""
        return self._get_workflow_list().setParent(self)

    def getHookManager(self):
        """@rtype: HookManager"""
        return self._get_hook_manager()

    def exportField(
        self, keyword, path, iactive, file_type, report_step, state, enkfFs
    ):
        """
        @type keyword: str
        @type path: str
        @type iactive: BoolVector
        @type file_type: EnkfFieldFileFormatEnum
        @type report_step: int
        @type enkfFs: EnkfFs

        """
        assert isinstance(keyword, str)
        return self._export_field_with_fs(
            keyword, path, iactive, file_type, report_step, state, enkfFs
        )

    def loadFromForwardModel(self, realization, iteration, fs):
        """Returns the number of loaded realizations"""
        return self._load_from_forward_model(iteration, realization, fs)

    def loadFromRunContext(self, run_context, fs):
        """Returns the number of loaded realizations"""
        return self._load_from_run_context(run_context, fs)

    def initRun(self, run_context):
        self._init_run(run_context)

    def createRunpath(self, run_context):
        self._create_run_path(run_context)

    def submitSimulation(self, run_arg, queue):
        self._submit_simulation(run_arg, queue)

    def getRunContextENSEMPLE_EXPERIMENT(self, fs, iactive, iteration=0):
        return self._alloc_run_context_ENSEMBLE_EXPERIMENT(fs, iactive, iteration)

    def create_runpath_list(self):
        return self._create_runpath_list()

    def getRunpathList(self):
        return self._get_runpath_list()

    def addNode(self, enkf_config_node):
        self._add_node(enkf_config_node)

    def rng(self):
        "Will return the random number generator used for updates."
        return self._get_shared_rng()
