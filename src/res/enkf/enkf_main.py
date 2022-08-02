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

from pathlib import Path
from typing import List

from cwrap import BaseCClass
from ecl.util.util import RandomNumberGenerator

from res import ResPrototype
from res._lib import enkf_main, enkf_state
from res.analysis.configuration import UpdateConfiguration
from res.enkf.analysis_config import AnalysisConfig
from res.enkf.ecl_config import EclConfig
from res.enkf.enkf_fs_manager import EnkfFsManager
from res.enkf.enkf_obs import EnkfObs
from res.enkf.enkf_simulation_runner import EnkfSimulationRunner
from res.enkf.ensemble_config import EnsembleConfig
from res.enkf.ert_run_context import RunContext
from res.enkf.ert_workflow_list import ErtWorkflowList
from res.enkf.hook_manager import HookManager
from res.enkf.key_manager import KeyManager
from res.enkf.model_config import ModelConfig
from res.enkf.queue_config import QueueConfig
from res.enkf.res_config import ResConfig
from res.enkf.runpaths import Runpaths
from res.enkf.site_config import SiteConfig
from res.enkf.substituter import Substituter
from res.util.substitution_list import SubstitutionList


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

    def __init__(self, config, strict=True, read_only=False):
        self.update_snapshots = {}
        real_enkf_main = _RealEnKFMain(config, strict, read_only)
        assert isinstance(real_enkf_main, BaseCClass)
        self._init_from_real_enkf_main(real_enkf_main)
        self._monkey_patch_methods(real_enkf_main)

        self._update_configuration = None

    def _init_from_real_enkf_main(self, real_enkf_main):
        super().__init__(
            real_enkf_main.from_param(real_enkf_main).value,
            parent=real_enkf_main,
            is_reference=True,
        )

        self.__simulation_runner = EnkfSimulationRunner(self)
        self.__fs_manager = EnkfFsManager(self)

    @property
    def update_configuration(self):
        if not self._update_configuration:
            global_update_step = [
                {
                    "name": "ALL_ACTIVE",
                    "observations": self._observation_keys,
                    "parameters": self._parameter_keys,
                }
            ]
            self._update_configuration = UpdateConfiguration(
                update_steps=global_update_step
            )
        return self._update_configuration

    @update_configuration.setter
    def update_configuration(self, user_config: List):
        config = UpdateConfiguration(update_steps=user_config)
        config.context_validate(self._observation_keys, self._parameter_keys)
        self._update_configuration = config

    @property
    def _observation_keys(self):
        return enkf_main.get_observation_keys(self)

    @property
    def _parameter_keys(self):
        return enkf_main.get_parameter_keys(self)

    @property
    def substituter(self):
        return self._real_enkf_main().substituter

    @property
    def runpaths(self):
        return self._real_enkf_main().runpaths

    @property
    def runpath_list_filename(self):
        return self._real_enkf_main().runpaths.runpath_list_filename

    def _real_enkf_main(self):
        return self.parent()

    def getEnkfSimulationRunner(self):
        """@rtype: EnkfSimulationRunner"""
        return self.__simulation_runner

    def getEnkfFsManager(self):
        """@rtype: EnkfFsManager"""
        return self.__fs_manager

    def umount(self):
        if self.__fs_manager is not None:
            self.__fs_manager.umount()
            self.__fs_manager = None

    def getLocalConfig(self) -> UpdateConfiguration:
        """@rtype: UpdateConfiguration"""
        return self.update_configuration

    def loadFromForwardModel(self, realization: List[bool], iteration: int, fs):
        """Returns the number of loaded realizations"""
        run_context = self.create_ensemble_experiment_run_context(
            active_mask=realization, iteration=iteration, source_filesystem=fs
        )
        nr_loaded = self.loadFromRunContext(run_context, fs)
        fs.sync()
        return nr_loaded

    def create_ensemble_experiment_run_context(
        self, iteration: int, active_mask: List[bool] = None, source_filesystem=None
    ) -> RunContext:
        """Creates an ensemble experiment run context
        :param fs: The source filesystem, defaults to
            getEnkfFsManager().getCurrentFileSystem().
        :param active_mask: Whether a realization is active or not,
            defaults to all active.
        """
        return self._create_run_context(
            iteration=iteration,
            active_mask=active_mask,
            source_filesystem=source_filesystem,
            target_fs=None,
        )

    def create_ensemble_smoother_run_context(
        self,
        iteration: int,
        target_filesystem,
        active_mask: List[bool] = None,
        source_filesystem=None,
    ) -> RunContext:
        """Creates an ensemble smoother run context
        :param fs: The source filesystem, defaults to
            getEnkfFsManager().getCurrentFileSystem().
        """
        return self._create_run_context(
            iteration=iteration,
            active_mask=active_mask,
            source_filesystem=source_filesystem,
            target_fs=target_filesystem,
        )

    def _create_run_context(
        self,
        iteration: int = 0,
        active_mask: List[bool] = None,
        source_filesystem=None,
        target_fs=None,
    ) -> RunContext:
        if active_mask is None:
            active_mask = [True] * self.getEnsembleSize()
        if source_filesystem is None:
            source_filesystem = self.getEnkfFsManager().getCurrentFileSystem()
        realizations = list(range(len(active_mask)))
        paths = self.runpaths.get_paths(realizations, iteration)
        jobnames = self.runpaths.get_jobnames(realizations, iteration)
        for realization, path in enumerate(paths):
            self.substituter.add_substitution("<RUNPATH>", path, realization, iteration)
        for realization, jobname in enumerate(jobnames):
            self.substituter.add_substitution(
                "<ECL_BASE>", jobname, realization, iteration
            )
            self.substituter.add_substitution(
                "<ECLBASE>", jobname, realization, iteration
            )
        return RunContext(
            sim_fs=source_filesystem,
            target_fs=target_fs,
            mask=active_mask,
            iteration=iteration,
            paths=paths,
            jobnames=jobnames,
        )

    def set_geo_id(self, geo_id: str, realization: int, iteration: int):
        self.substituter.add_substitution("<GEO_ID>", geo_id, realization, iteration)

    def write_runpath_list(self, iterations: List[int], realizations: List[int]):
        self.runpaths.write_runpath_list(iterations, realizations)

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
    - EnkfSimulationRunner, EnkfFsManager: access specific
      functionalities
    EnKFMain owns an instance of each of the last 3 classes. Also, all
    of these classes need to access the same underlying C object.
    So, in order to avoid circular dependencies, we make _RealEnKF main
    the only "owner" of the C object, and all the classes that need to
    access it set _RealEnKFMain as parent.

    The situation can be summarized as follows (show only EnkfFSManager,
    EnkfSimulationRunner are treated analogously)
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

    _alloc = ResPrototype("void* enkf_main_alloc(res_config, bool)", bind=False)

    _free = ResPrototype("void enkf_main_free(enkf_main)")
    _get_ensemble_size = ResPrototype("int enkf_main_get_ensemble_size( enkf_main )")
    _get_data_kw = ResPrototype("subst_list_ref enkf_main_get_data_kw(enkf_main)")
    _get_obs = ResPrototype("enkf_obs_ref enkf_main_get_obs(enkf_main)")
    _get_observations = ResPrototype(
        "void enkf_main_get_observations(enkf_main, \
                                         char*, \
                                         int, \
                                         long*, \
                                         double*, \
                                         double*)"
    )
    _have_observations = ResPrototype("bool enkf_main_have_obs(enkf_main)")
    _get_workflow_list = ResPrototype(
        "ert_workflow_list_ref enkf_main_get_workflow_list(enkf_main)"
    )
    _get_hook_manager = ResPrototype(
        "hook_manager_ref enkf_main_get_hook_manager(enkf_main)"
    )
    _get_res_config = ResPrototype("res_config_ref enkf_main_get_res_config(enkf_main)")
    _get_shared_rng = ResPrototype("rng_ref enkf_main_get_shared_rng(enkf_main)")

    def __init__(self, config, strict=True, read_only=False):
        """Please don't use this class directly. See EnKFMain instead"""
        self.config_file = config
        res_config = self._init_res_config(config)
        if res_config is None:
            raise TypeError(
                "Failed to construct EnKFMain instance due to invalid res_config."
            )

        c_ptr = self._alloc(res_config, read_only)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                f"Failed to construct EnKFMain instance from config {res_config}."
            )

        self.__key_manager = KeyManager(self)
        self.substituter = Substituter(
            {key: value for (key, value, _) in self.getDataKW()}
        )
        self.runpaths = Runpaths(
            self.getModelConfig().getJobnameFormat(),
            self.getModelConfig().getRunpathFormat().format_string,
            Path(res_config.runpath_file),
            self.substituter.substitute,
        )

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

        raise TypeError(f"Expected ResConfig, received: {repr(config)}")

    def get_queue_config(self) -> QueueConfig:
        return self.resConfig().queue_config

    def free(self):
        self._free()

    def __repr__(self):
        ens = self.getEnsembleSize()
        cnt = f"ensemble_size = {ens}, config_file = {self.config_file}"
        return self._create_repr(cnt)

    def getEnsembleSize(self) -> int:
        return self._get_ensemble_size()

    def ensembleConfig(self) -> EnsembleConfig:
        return self.resConfig().ensemble_config.setParent(self)

    def analysisConfig(self) -> AnalysisConfig:
        return self.resConfig().analysis_config.setParent(self)

    def getModelConfig(self) -> ModelConfig:
        return self.resConfig().model_config.setParent(self)

    def siteConfig(self) -> SiteConfig:
        return self.resConfig().site_config.setParent(self)

    def resConfig(self) -> "ResConfig":
        return self._get_res_config().setParent(self)

    def eclConfig(self) -> EclConfig:
        return self.resConfig().ecl_config

    def getDataKW(self) -> SubstitutionList:
        return self._get_data_kw()

    def addDataKW(self, key, value):
        # Substitution should be the responsibility of
        # self.substituter. However,
        # self.resConfig().subst_config.subst_list is still
        # used by workflows to do substitution. For now, we
        # need to update this here.
        self.resConfig().subst_config.subst_list.addItem(key, value)
        self.substituter.add_global_substitution(key, value)

    def getMountPoint(self) -> str:
        return self.resConfig().model_config.getEnspath()

    def getObservations(self) -> EnkfObs:
        """@rtype: EnkfObs"""
        return self._get_obs().setParent(self)

    def have_observations(self):
        return self._have_observations()

    def getHistoryLength(self) -> int:
        return self.resConfig().model_config.get_last_history_restart()

    def get_observations(self, user_key, obs_count, obs_x, obs_y, obs_std):
        return self._get_observations(user_key, obs_count, obs_x, obs_y, obs_std)

    def getKeyManager(self) -> "KeyManager":
        return self.__key_manager

    def getWorkflowList(self) -> ErtWorkflowList:
        return self._get_workflow_list().setParent(self)

    def getHookManager(self) -> HookManager:
        return self._get_hook_manager()

    def loadFromRunContext(self, run_context: RunContext, fs) -> int:
        """Returns the number of loaded realizations"""
        return enkf_main.load_from_run_context(
            self, run_context.run_args, run_context.mask, fs
        )

    def initRun(self, run_context):
        enkf_main.init_internalization(self)

        for realization_nr in range(self.getEnsembleSize()):
            if run_context.is_active(realization_nr):
                enkf_state.state_initialize(
                    self,
                    run_context.sim_fs,
                    self._parameter_keys,
                    run_context.init_mode.value,
                    realization_nr,
                )

    def rng(self) -> RandomNumberGenerator:
        "Will return the random number generator used for updates."
        return self._get_shared_rng()

    @property
    def _parameter_keys(self):
        return enkf_main.get_parameter_keys(self)
