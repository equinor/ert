import io
import logging
import os
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, Union

import numpy as np
import pandas as pd

from ert import _clib
from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.enkf import EnkfFs
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.data import EnkfNode
from ert._c_wrappers.enkf.enkf_fs_manager import FileSystemManager
from ert._c_wrappers.enkf.enkf_obs import EnkfObs
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert._c_wrappers.enkf.ert_workflow_list import ErtWorkflowList
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.node_id import NodeId
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.enkf.runpaths import Runpaths
from ert._c_wrappers.enkf.site_config import SiteConfig
from ert._c_wrappers.enkf.substituter import Substituter
from ert._c_wrappers.util.substitution_list import SubstitutionList
from ert._clib.state_map import (
    STATE_LOAD_FAILURE,
    STATE_PARENT_FAILURE,
    STATE_UNDEFINED,
)

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.res_config import ResConfig
    from ert._c_wrappers.enkf.state_map import StateMap


logger = logging.getLogger(__name__)


def naturalSortKey(s: str) -> List[Union[int, str]]:
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


class EnKFMain:
    def __init__(self, config: "ResConfig", read_only: bool = False):
        self.res_config = config
        self.update_snapshots = {}
        self._update_configuration = None
        if config is None:
            raise TypeError(
                "Failed to construct EnKFMain instance due to invalid res_config."
            )

        self._observations = EnkfObs(
            config.model_config.get_history_source(),
            config.model_config.get_time_map(),
            config.ensemble_config.grid,
            config.ensemble_config.refcase,
            config.ensemble_config,
        )
        if config.model_config.obs_config_file:
            self._observations.load(
                config.model_config.obs_config_file,
                config.analysis_config.get_std_cutoff(),
            )
        self._ensemble_size = self.res_config.model_config.num_realizations
        self._substituter = Substituter(
            {key: value for (key, value, _) in self.getDataKW()}
        )
        self._runpaths = Runpaths(
            self.getModelConfig().getJobnameFormat(),
            self.getModelConfig().getRunpathFormat().format_string,
            Path(config.runpath_file),
            self.substituter.substitute,
        )

        # Initialize storage
        ens_path = Path(config.model_config.getEnspath())
        self.storage_manager = FileSystemManager(
            5,
            ens_path,
            config.ensemble_config,
            self.getEnsembleSize(),
            read_only=read_only,
        )
        self.switchFileSystem(self.storage_manager.active_case)

        # Set up RNG
        config_seed = self.resConfig().random_seed
        if config_seed is None:
            seed_seq = np.random.SeedSequence()
            logger.info(
                "To repeat this experiment, "
                "add the following random seed to your config file:"
            )
            logger.info(f"RANDOM_SEED {seed_seq.entropy}")
        else:
            seed: Union[int, Sequence[int]]
            try:
                seed = int(config_seed)
            except ValueError:
                seed = [ord(x) for x in config_seed]
            seed_seq = np.random.SeedSequence(seed)

        self._shared_rng = np.random.default_rng(seed_seq)

        self.realizations = [
            np.random.default_rng(seed)
            for seed in seed_seq.spawn(self.getEnsembleSize())
        ]

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

    def getEnkfFsManager(self) -> "EnKFMain":
        return self

    def getLocalConfig(self) -> "UpdateConfiguration":
        return self.update_configuration

    def loadFromForwardModel(
        self, realization: List[bool], iteration: int, fs: "EnkfFs"
    ) -> int:
        """Returns the number of loaded realizations"""
        run_context = self.create_ensemble_experiment_run_context(
            active_mask=realization, iteration=iteration, source_filesystem=fs
        )
        nr_loaded = fs.load_from_run_path(
            self.getEnsembleSize(),
            self.ensembleConfig(),
            self.getModelConfig(),
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

    def get_num_cpu(self) -> int:
        return self.res_config.preferred_num_cpu()

    def __repr__(self):
        return f"EnKFMain(size: {self.getEnsembleSize()}, config: {self.res_config})"

    def getEnsembleSize(self) -> int:
        return self._ensemble_size

    def ensembleConfig(self) -> EnsembleConfig:
        return self.resConfig().ensemble_config

    def analysisConfig(self) -> AnalysisConfig:
        return self.resConfig().analysis_config

    def getModelConfig(self) -> ModelConfig:
        return self.res_config.model_config

    def siteConfig(self) -> SiteConfig:
        return self.res_config.site_config

    def resConfig(self) -> "ResConfig":
        return self.res_config

    def getDataKW(self) -> SubstitutionList:
        return self.res_config.subst_config.subst_list

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

    def getWorkflowList(self) -> ErtWorkflowList:
        return self.resConfig().ert_workflow_list

    def initRun(self, run_context: "RunContext", parameters: List[str] = None):
        # pylint: disable=too-many-nested-blocks
        # (this is a real code smell that we mute for now)
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
                    config_node = self.ensembleConfig().getNode(parameter)
                    enkf_node = EnkfNode(config_node)

                    if config_node.getUseForwardInit():
                        continue
                    if (
                        not enkf_node.has_data(
                            run_context.sim_fs, NodeId(0, realization_nr)
                        )
                        or current_status == STATE_LOAD_FAILURE
                    ):
                        rng = self.realizations[realization_nr]
                        impl_type = enkf_node.getImplType()

                        if impl_type == ErtImplType.GEN_KW:
                            gen_kw_node = enkf_node.asGenKw()
                            if len(gen_kw_node) > 0:
                                if config_node.get_init_file_fmt():
                                    df = pd.read_csv(
                                        config_node.get_init_file_fmt()
                                        % realization_nr,
                                        delim_whitespace=True,
                                        header=None,
                                    )
                                    # This means we have a key: value mapping in the
                                    # file otherwise it is just a list of values
                                    if df.shape[1] == 2:
                                        # We need to sort the user input keys by the
                                        # internal order of sub-parameters:
                                        keys = [val[0] for val in gen_kw_node.items()]
                                        df = df.set_index(df.columns[0])
                                        vals = df.reindex(keys).values.flatten()
                                    else:
                                        vals = df.values.flatten()
                                else:
                                    vals = rng.standard_normal(len(gen_kw_node))
                                s = io.BytesIO()
                                # The first element is time_t (64 bit integer), but
                                # it is not used so we write 0 instead - for
                                # padding purposes.
                                s.write(struct.pack("Qi", 0, int(ErtImplType.GEN_KW)))
                                s.write(vals.tobytes())

                                _clib.enkf_fs.write_parameter(
                                    run_context.sim_fs,
                                    config_node.getKey(),
                                    realization_nr,
                                    s.getvalue(),
                                )

                        else:
                            _clib.enkf_state.state_initialize(
                                enkf_node,
                                run_context.sim_fs,
                                realization_nr,
                            )

                if state_map[realization_nr] in [STATE_UNDEFINED, STATE_LOAD_FAILURE]:
                    state_map[realization_nr] = RealizationStateEnum.STATE_INITIALIZED
        run_context.sim_fs.sync()

    def rng(self) -> np.random.Generator:
        "Will return the random number generator used for updates."
        return self._shared_rng

    def getFileSystem(self, case_name: str) -> "EnkfFs":
        try:
            case = self.storage_manager[case_name]
        except KeyError:
            case = self.storage_manager.add_case(case_name)
        if self.res_config.ensemble_config.refcase:
            time_map = case.getTimeMap()
            time_map.attach_refcase(self.res_config.ensemble_config.refcase)
        return case

    def caseExists(self, case_name: str) -> bool:
        return case_name in self.storage_manager

    def caseHasData(self, case_name: str) -> bool:
        if case_name not in self.storage_manager:
            return False
        state_map = self.storage_manager.state_map(case_name)

        return any(state == RealizationStateEnum.STATE_HAS_DATA for state in state_map)

    def getCurrentFileSystem(self) -> "EnkfFs":
        """Returns the currently selected file system"""
        return self.getFileSystem(self.storage_manager.active_case)

    def getStateMapForCase(self, case_name: str):
        return self.storage_manager.state_map(case_name)

    def switchFileSystem(self, case_name: str) -> None:
        if isinstance(case_name, EnkfFs):
            case_name = case_name.case_name
        if case_name not in self.storage_manager.cases:
            raise KeyError(
                f"Unknown case: {case_name}, valid: {self.storage_manager.cases}"
            )
        self.addDataKW("<ERT-CASE>", case_name)
        self.addDataKW("<ERTCASE>", case_name)
        self.storage_manager.active_case = case_name
        (Path(self.getModelConfig().getEnspath()) / "current_case").write_text(
            case_name
        )

    def isCaseInitialized(self, case: str) -> bool:
        if case not in self.storage_manager:
            return False
        return self.storage_manager[case].is_initalized

    def getCaseList(self) -> List[str]:
        return sorted(self.storage_manager.cases, key=naturalSortKey)

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

                for source_file, target_file in self.res_config.ert_templates:
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

                res_config = self.resConfig()
                model_config = res_config.model_config
                _clib.enkf_main.ecl_write(
                    model_config=model_config,
                    ensemble_config=res_config.ensemble_config,
                    run_path=run_arg.runpath,
                    iens=run_arg.iens,
                    sim_fs=run_arg.sim_fs,
                )
                model_config.getForwardModel().formatted_fprintf(
                    run_arg.get_run_id(),
                    run_arg.runpath,
                    model_config.data_root(),
                    subst_list,
                    res_config.site_config.env_vars,
                )

        active_list = [
            run_context[i] for i in range(len(run_context)) if run_context.is_active(i)
        ]
        iterations = sorted({runarg.iter_id for runarg in active_list})
        realizations = sorted({runarg.iens for runarg in active_list})

        self.runpaths.write_runpath_list(iterations, realizations)

    def runWorkflows(self, runtime: int) -> None:
        workflow_list = self.getWorkflowList()
        for workflow in workflow_list.get_workflows_hooked_at(runtime):
            workflow.run(self, context=workflow_list.getContext())
