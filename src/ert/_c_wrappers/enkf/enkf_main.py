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
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

from ecl.util.enums import RngInitModeEnum
from ecl.util.util import RandomNumberGenerator

from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.data import EnkfNode
from ert._c_wrappers.enkf.ecl_config import EclConfig
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.enkf_fs_manager import FileSystemRotator
from ert._c_wrappers.enkf.enkf_obs import EnkfObs
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert._c_wrappers.enkf.ert_workflow_list import ErtWorkflowList
from ert._c_wrappers.enkf.hook_manager import HookManager
from ert._c_wrappers.enkf.key_manager import KeyManager
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.node_id import NodeId
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.enkf.runpaths import Runpaths
from ert._c_wrappers.enkf.site_config import SiteConfig
from ert._c_wrappers.enkf.substituter import Substituter
from ert._c_wrappers.util.substitution_list import SubstitutionList
from ert._clib import enkf_main, enkf_state
from ert._clib.state_map import (
    STATE_LOAD_FAILURE,
    STATE_PARENT_FAILURE,
    STATE_UNDEFINED,
)

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.res_config import ResConfig
    from ert._c_wrappers.enkf.state_map import StateMap


def naturalSortKey(s: str) -> List[Union[int, str]]:
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def _forward_rng(rng):
    """
    The rng state needs a byte string of length 16, to get that
    we forward, i.e. sample from the rng 4 times and convert to
    byte, creating a byte string of length 16.
    """
    return b"".join(struct.pack("I", (rng.forward())) for _ in range(4))


def format_seed(random_seed: str):
    state_size = 4
    state_digits = 10
    fseed = [0] * state_size
    seed_pos = 0
    for i in range(state_size):
        for _ in range(state_digits):
            fseed[i] *= 10
            fseed[i] += ord(random_seed[seed_pos]) - ord("0")
            seed_pos = (seed_pos + 1) % len(random_seed)

    # The function this was derived from had integer overflow, so we
    # allow for the same here
    return b"".join(struct.pack("I", x % (2**32)) for x in fseed)


class EnKFMain:
    def __init__(self, config: "ResConfig", read_only: bool = False):
        self.config_file = config
        self.update_snapshots = {}
        self._update_configuration = None
        if config is None:
            raise TypeError(
                "Failed to construct EnKFMain instance due to invalid res_config."
            )

        self._observations = EnkfObs(
            config.model_config.get_history_source(),
            config.model_config.get_time_map(),
            config.ecl_config.getGrid(),
            config.ecl_config.getRefcase(),
            config.ensemble_config,
        )
        if config.model_config.obs_config_file:
            self._observations.load(
                config.model_config.obs_config_file,
                config.analysis_config.getStdCutoff(),
            )
        self._ensemble_size = self.config_file.model_config.num_realizations
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
        global_rng = RandomNumberGenerator()
        self._shared_rng = RandomNumberGenerator(init_mode=RngInitModeEnum.INIT_DEFAULT)
        random_seed = self.resConfig().random_seed
        if random_seed:
            global_rng.setState(format_seed(random_seed))
            self._shared_rng.setState(format_seed(random_seed))
            self._shared_rng.setState(_forward_rng(self._shared_rng))
            self._shared_rng.setState(_forward_rng(self._shared_rng))
        else:
            enkf_main.log_seed(global_rng)
        self.realizations = [
            RandomNumberGenerator(init_mode=RngInitModeEnum.INIT_DEFAULT)
            for _ in range(self.getEnsembleSize())
        ]
        for rng in self.realizations:
            rng.setState(_forward_rng(global_rng))

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
        return list(self._observations.getMatchingKeys("*"))

    @property
    def _parameter_keys(self):
        return self.ensembleConfig().parameters

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
        nr_loaded = fs.load_from_run_path(
            self.getEnsembleSize(),
            self.ensembleConfig(),
            self.getModelConfig(),
            self.eclConfig(),
            run_context.run_args,
            run_context.mask,
        )
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

    def __repr__(self):
        return f"EnKFMain(size: {self.getEnsembleSize()}, config: {self.config_file})"

    def getEnsembleSize(self) -> int:
        return self._ensemble_size

    def ensembleConfig(self) -> EnsembleConfig:
        return self.resConfig().ensemble_config

    def analysisConfig(self) -> AnalysisConfig:
        return self.resConfig().analysis_config

    def getModelConfig(self) -> ModelConfig:
        return self.config_file.model_config

    def siteConfig(self) -> SiteConfig:
        return self.config_file.site_config

    def resConfig(self) -> "ResConfig":
        return self.config_file

    def eclConfig(self) -> EclConfig:
        return self.config_file.ecl_config

    def getDataKW(self) -> SubstitutionList:
        return self.config_file.subst_config.subst_list

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
        return self._observations

    def have_observations(self) -> bool:
        return len(self._observations) > 0

    def getHistoryLength(self) -> int:
        return self.resConfig().model_config.get_last_history_restart()

    def getKeyManager(self) -> "KeyManager":
        return self.__key_manager

    def getWorkflowList(self) -> ErtWorkflowList:
        return self.resConfig().ert_workflow_list

    def getHookManager(self) -> HookManager:
        return self.resConfig().hook_manager

    def initRun(self, run_context: "RunContext", parameters: List[str] = None):
        if parameters is None:
            parameters = self._parameter_keys
        state_map = run_context.sim_fs.getStateMap()
        for realization_nr, rng in enumerate(self.realizations):
            current_status = state_map[realization_nr]
            if (
                run_context.is_active(realization_nr)
                and not current_status == STATE_PARENT_FAILURE
            ):
                for parameter in parameters:
                    node = self.ensembleConfig().getNode(parameter)
                    enkf_node = EnkfNode(node)
                    if node.getUseForwardInit():
                        continue
                    if (
                        not enkf_node.has_data(
                            run_context.sim_fs, NodeId(0, realization_nr)
                        )
                        or current_status == STATE_LOAD_FAILURE
                    ):
                        enkf_state.state_initialize(
                            rng,
                            enkf_node,
                            run_context.sim_fs,
                            realization_nr,
                        )
                if state_map[realization_nr] in [STATE_UNDEFINED, STATE_LOAD_FAILURE]:
                    state_map[realization_nr] = RealizationStateEnum.STATE_INITIALIZED
        run_context.sim_fs.sync()

    def rng(self) -> RandomNumberGenerator:
        "Will return the random number generator used for updates."
        return self._shared_rng

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
                    model_config=self.resConfig().model_config,
                    ensemble_config=self.resConfig().ensemble_config,
                    site_config=self.resConfig().site_config,
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
