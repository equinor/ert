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
import os
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

from cwrap import BaseCClass
from ecl.util.util import RandomNumberGenerator

from ert._c_wrappers import ResPrototype
from ert._clib import enkf_main, enkf_state, model_callbacks
from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.enkf.enums import RealizationStateEnum, ErtImplType
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.ecl_config import EclConfig
from ert._c_wrappers.enkf.enkf_fs_manager import FileSystemRotator
from ert._c_wrappers.enkf.enkf_obs import EnkfObs
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert._c_wrappers.enkf.ert_workflow_list import ErtWorkflowList
from ert._c_wrappers.enkf.hook_manager import HookManager
from ert._c_wrappers.enkf.key_manager import KeyManager
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.enkf.runpaths import Runpaths
from ert._c_wrappers.enkf.site_config import SiteConfig
from ert._c_wrappers.enkf.substituter import Substituter
from ert._c_wrappers.job_queue import JobQueueManager, RunStatusType
from ert._c_wrappers.util.substitution_list import SubstitutionList

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.state_map import StateMap
    from ert._c_wrappers.job_queue.queue import JobQueue
    from ert._c_wrappers.enkf.hook_manager import HookRuntime
    from ert._c_wrappers.enkf.res_config import ResConfig


def naturalSortKey(s: str) -> List[Union[int, str]]:
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


class EnKFMain(BaseCClass):
    """Access to the C EnKFMain interface."""

    TYPE_NAME = "enkf_main"

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

    def __init__(
        self, config: "ResConfig", strict: bool = True, read_only: bool = False
    ):
        self.config_file = config
        self.update_snapshots = {}
        self._update_configuration = None
        if config is None:
            raise TypeError(
                "Failed to construct EnKFMain instance due to invalid res_config."
            )
        c_ptr = self._alloc(config, read_only)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                f"Failed to construct EnKFMain instance from config {config}."
            )
        self.__key_manager = KeyManager(self)
        self._substituter = Substituter(
            {key: value for (key, value, _) in self.getDataKW()}
        )
        self._runpaths = Runpaths(
            self.getModelConfig().getJobnameFormat(),
            self.getModelConfig().getRunpathFormat().format_string,
            Path(config.runpath_file),
            self.substituter.substitute,
        )

        # Initalize storage
        self._fs_rotator = FileSystemRotator(5)
        ens_path = Path(config.model_config.getEnspath())
        current_case_file = ens_path / "current_case"
        if current_case_file.exists():
            fs = EnkfFs(
                ens_path / current_case_file.read_text("utf-8").strip(),
                read_only=read_only,
            )
        else:
            fs = EnkfFs.createFileSystem(ens_path / "default", read_only=read_only)
        self.storage = fs

    @property
    def update_configuration(self) -> UpdateConfiguration:
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
    def update_configuration(self, user_config: Any):
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
        return self._substituter

    @property
    def runpaths(self):
        return self._runpaths

    @property
    def runpath_list_filename(self):
        return self._runpaths.runpath_list_filename

    @property
    def storage(self):
        return self._fs

    @storage.setter
    def storage(self, file_system):
        self.addDataKW("<ERT-CASE>", file_system.getCaseName())
        self.addDataKW("<ERTCASE>", file_system.getCaseName())
        for key in file_system.getSummaryKeySet().keys():
            self.ensembleConfig().add_summary(key)
        if self.config_file.ecl_config.getRefcase():
            time_map = file_system.getTimeMap()
            time_map.attach_refcase(self.config_file.ecl_config.getRefcase())
        case_name = file_system.getCaseName()
        full_name = self._createFullCaseName(self.getMountPoint(), case_name)
        if full_name not in self._fs_rotator:
            self._fs_rotator.append(file_system)
        # On setting a new file system we write the current_case file
        (Path(self.getModelConfig().getEnspath()) / "current_case").write_text(
            file_system.getCaseName()
        )
        self._fs = file_system

    def getEnkfFsManager(self) -> "EnKFMain":
        return self

    def getLocalConfig(self) -> "UpdateConfiguration":
        return self.update_configuration

    def loadFromForwardModel(
        self, realization: List[bool], iteration: int, fs: EnkfFs
    ) -> int:
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
            source_filesystem = self.getCurrentFileSystem()
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

    def addDataKW(self, key: str, value: str) -> None:
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
        return self._get_obs().setParent(self)

    def have_observations(self) -> bool:
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

    def _createFullCaseName(self, mount_root: str, case_name: str) -> str:
        return os.path.join(mount_root, case_name)

    # The return value from the getFileSystem will be a weak reference to the
    # underlying enkf_fs object. That implies that the fs manager must be in
    # scope for the return value to be valid.
    def getFileSystem(
        self, case_name: str, mount_root: str = None, read_only: bool = False
    ) -> EnkfFs:
        if mount_root is None:
            mount_root = self.getMountPoint()

        full_case_name = self._createFullCaseName(mount_root, case_name)

        if full_case_name not in self._fs_rotator:
            if not os.path.exists(full_case_name):
                new_fs = EnkfFs.createFileSystem(full_case_name, read_only)
            else:
                new_fs = EnkfFs(full_case_name, read_only)
            self._fs_rotator.append(new_fs)

        fs = self._fs_rotator[full_case_name]

        return fs

    def caseExists(self, case_name: str) -> bool:
        return case_name in self.getCaseList()

    def caseHasData(self, case_name: str) -> bool:
        state_map = self.getStateMapForCase(case_name)

        return any(state == RealizationStateEnum.STATE_HAS_DATA for state in state_map)

    def getCurrentFileSystem(self) -> EnkfFs:
        """Returns the currently selected file system"""
        return self.storage

    def getFileSystemCount(self) -> int:
        return len(self._fs_rotator)

    def switchFileSystem(self, file_system: EnkfFs) -> None:
        self.storage = file_system

    def isCaseInitialized(self, case: str) -> bool:
        case = os.path.join(self.getMountPoint(), case)
        if case not in self._fs_rotator:
            return False
        return self._fs_rotator[case].is_initalized(
            self.ensembleConfig(), self._parameter_keys, self.getEnsembleSize()
        )

    def getCaseList(self) -> List[str]:
        caselist = [
            str(x.stem) for x in Path(self.getMountPoint()).iterdir() if x.is_dir()
        ]
        return sorted(caselist, key=naturalSortKey)

    def customInitializeCurrentFromExistingCase(
        self,
        source_case: str,
        source_report_step: int,
        member_mask: List[bool],
        node_list: List[str],
    ) -> None:
        if source_case not in self.getCaseList():
            raise KeyError(
                f"No such source case: {source_case} in {self.getCaseList()}"
            )
        source_case_fs = self.getFileSystem(source_case)
        enkf_main.init_current_case_from_existing_custom(
            self.ensembleConfig(),
            source_case_fs,
            self._fs,
            source_report_step,
            node_list,
            member_mask,
        )

    def initializeFromScratch(
        self, parameter_list: List[str], run_context: RunContext
    ) -> None:
        for realization_nr in range(self.getEnsembleSize()):
            if run_context.is_active(realization_nr):
                enkf_state.state_initialize(
                    self,
                    run_context.sim_fs,
                    parameter_list,
                    run_context.init_mode.value,
                    realization_nr,
                )

    def isCaseMounted(self, case_name: str, mount_root: str = None) -> bool:
        if mount_root is None:
            mount_root = self.getMountPoint()

        full_case_name = self._createFullCaseName(mount_root, case_name)

        return full_case_name in self._fs_rotator

    def getStateMapForCase(self, case: str) -> "StateMap":
        if self.isCaseMounted(case):
            fs = self.getFileSystem(case)
            return fs.getStateMap()
        else:
            mount_root = self.getMountPoint()
            full_case_name = self._createFullCaseName(mount_root, case)
            return self.storage.read_state_map(full_case_name)

    def createRunPath(self, run_context: RunContext) -> None:
        self.initRun(run_context)
        for iens, run_arg in enumerate(run_context):
            if run_context.is_active(iens):
                substitutions = self.substituter.get_substitutions(
                    iens, run_arg.iter_id
                )
                subst_list = SubstitutionList()
                for subst in substitutions.items():
                    subst_list.addItem(*subst)
                os.makedirs(
                    run_arg.runpath,
                    exist_ok=True,
                )

                for source_file, target_file in self.config_file.ert_templates:
                    target_file = self.substituter.substitute(
                        target_file, run_arg.iens, run_context.iteration
                    )
                    result = self.substituter.substitute(
                        Path(source_file).read_text("utf-8"),
                        run_arg.iens,
                        run_context.iteration,
                    )
                    target = Path(run_arg.runpath) / target_file
                    if not target.parent.exists():
                        os.makedirs(
                            target.parent,
                            exist_ok=True,
                        )
                    target.write_text(result)

                enkf_main.init_active_run(
                    res_config=self.resConfig(),
                    run_path=run_arg.runpath,
                    iens=run_arg.iens,
                    sim_fs=run_arg.sim_fs,
                    run_id=run_arg.get_run_id(),
                    job_name=run_arg.job_name,
                    subst_list=subst_list,
                )

        active_list = [
            run_context[i] for i in range(len(run_context)) if run_context.is_active(i)
        ]
        iterations = sorted({runarg.iter_id for runarg in active_list})
        realizations = sorted({runarg.iens for runarg in active_list})

        self.runpaths.write_runpath_list(iterations, realizations)

    @staticmethod
    def runWorkflows(runtime: int, ert: "EnKFMain") -> None:
        hook_manager = ert.getHookManager()
        hook_manager.runWorkflows(runtime, ert)
