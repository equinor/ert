from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from numpy.random import SeedSequence

from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.enkf_obs import EnkfObs
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.util.substitution_list import SubstitutionList
from ert.job_queue import WorkflowRunner
from ert.runpaths import Runpaths

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import ErtConfig
    from ert._c_wrappers.enkf.enums import HookRuntime
    from ert._c_wrappers.enkf.queue_config import QueueConfig
    from ert.storage import EnsembleAccessor, EnsembleReader, StorageAccessor

logger = logging.getLogger(__name__)


def _backup_if_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%SZ")
    new_path = path.parent / f"{path.name}_backup_{timestamp}"
    path.rename(new_path)


def _value_export_txt(
    run_path: Path, export_base_name: str, values: Mapping[str, Mapping[str, float]]
) -> None:
    path = run_path / f"{export_base_name}.txt"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    with path.open("w") as f:
        for key, param_map in values.items():
            for param, value in param_map.items():
                print(f"{key}:{param} {value:g}", file=f)


def _value_export_json(
    run_path: Path, export_base_name: str, values: Mapping[str, Mapping[str, float]]
) -> None:
    path = run_path / f"{export_base_name}.json"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    # Hierarchical
    json_out: Dict[str, Union[float, Dict[str, float]]] = {
        key: dict(param_map.items()) for key, param_map in values.items()
    }

    # Composite
    json_out.update(
        {
            f"{key}:{param}": value
            for key, param_map in values.items()
            for param, value in param_map.items()
        }
    )

    # Disallow NaN from being written: ERT produces the parameters and the only
    # way for the output to be NaN is if the input is invalid or if the sampling
    # function is buggy. Either way, that would be a bug and we can report it by
    # having json throw an error.
    json.dump(
        json_out, path.open("w"), allow_nan=False, indent=0, separators=(", ", " : ")
    )


def _generate_parameter_files(
    parameter_configs: Iterable[ParameterConfig],
    export_base_name: str,
    run_path: Path,
    iens: int,
    fs: EnsembleReader,
    iteration: int,
) -> None:
    """
    Generate parameter files that are placed in each runtime directory for
    forward-model jobs to consume.

    Args:
        parameter_configs: Configuration which contains the parameter nodes for this
            ensemble run.
        export_base_name: Base name for the GEN_KW parameters file. Ie. the
            `parameters` in `parameters.json`.
        run_path: Path to the runtime directory
        iens: Realisation index
        fs: EnsembleReader from which to load parameter data
    """
    exports: Dict[str, Dict[str, float]] = {}

    for node in parameter_configs:
        # For the first iteration we do not write the parameter
        # to run path, as we expect to read if after the forward
        # model has completed.
        if node.forward_init and iteration == 0:
            continue
        export_values = node.write_to_runpath(Path(run_path), iens, fs)
        if export_values:
            exports.update(export_values)
        continue

    _value_export_txt(run_path, export_base_name, exports)
    _value_export_json(run_path, export_base_name, exports)


def _seed_sequence(seed: Optional[str]) -> SeedSequence:
    # Set up RNG
    if seed is None:
        sequence = SeedSequence()
        logger.info(
            "To repeat this experiment, "
            "add the following random seed to your config file:"
        )
        logger.info(f"RANDOM_SEED {sequence.entropy}")
        return sequence

    int_seed: Optional[Union[int, Sequence[int]]] = None
    try:
        int_seed = int(seed)
    except ValueError:
        int_seed = [ord(x) for x in seed]
    return SeedSequence(int_seed)


class EnKFMain:
    def __init__(self, config: "ErtConfig", read_only: bool = False) -> None:
        self.ert_config = config
        self._update_configuration: Optional[UpdateConfiguration] = None

        self._observations = EnkfObs.from_ert_config(config)

        self._ensemble_size = self.ert_config.model_config.num_realizations
        self._runpaths = Runpaths(
            jobname_format=self.getModelConfig().jobname_format_string,
            runpath_format=self.getModelConfig().runpath_format_string,
            filename=str(self.ert_config.runpath_file),
            substitute=self.get_context().substitute_real_iter,
        )

        self._global_seed = _seed_sequence(self.ert_config.random_seed)
        self._shared_rng = np.random.default_rng(self._global_seed)

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
    def update_configuration(self, user_config: Any) -> None:
        config = UpdateConfiguration(update_steps=user_config)
        config.context_validate(self._observation_keys, self._parameter_keys)
        self._update_configuration = config

    @property
    def _observation_keys(self) -> List[str]:
        return list(self._observations.getMatchingKeys("*"))

    @property
    def _parameter_keys(self) -> List[str]:
        return self.ensembleConfig().parameters

    @property
    def runpaths(self) -> Runpaths:
        return self._runpaths

    @property
    def runpath_list_filename(self) -> os.PathLike[str]:
        return self._runpaths.runpath_list_filename

    def getLocalConfig(self) -> "UpdateConfiguration":
        return self.update_configuration

    def loadFromForwardModel(
        self, realization: List[bool], iteration: int, fs: EnsembleAccessor
    ) -> int:
        """Returns the number of loaded realizations"""
        t = time.perf_counter()
        run_context = self.ensemble_context(fs, realization, iteration)
        nr_loaded = fs.load_from_run_path(
            self.getEnsembleSize(),
            self.ensembleConfig(),
            run_context.run_args,
            run_context.mask,
        )
        fs.sync()
        logger.debug(
            f"loadFromForwardModel() time_used {(time.perf_counter() - t):.4f}s"
        )
        return nr_loaded

    def ensemble_context(
        self, case: EnsembleAccessor, active_realizations: List[bool], iteration: int
    ) -> RunContext:
        """This loads an existing case from storage
        and creates run information for that case"""
        self.addDataKW("<ERT-CASE>", case.name)
        self.addDataKW("<ERTCASE>", case.name)
        return RunContext(
            sim_fs=case,
            runpaths=self._runpaths,
            initial_mask=active_realizations,
            iteration=iteration,
        )

    def write_runpath_list(
        self, iterations: List[int], realizations: List[int]
    ) -> None:
        self.runpaths.write_runpath_list(iterations, realizations)

    def get_queue_config(self) -> QueueConfig:
        return self.resConfig().queue_config

    def get_num_cpu(self) -> int:
        return self.ert_config.preferred_num_cpu()

    def __repr__(self) -> str:
        return f"EnKFMain(size: {self.getEnsembleSize()}, config: {self.ert_config})"

    def getEnsembleSize(self) -> int:
        return self._ensemble_size

    def ensembleConfig(self) -> EnsembleConfig:
        return self.resConfig().ensemble_config

    def analysisConfig(self) -> AnalysisConfig:
        return self.resConfig().analysis_config

    def getModelConfig(self) -> ModelConfig:
        return self.ert_config.model_config

    def resConfig(self) -> "ErtConfig":
        return self.ert_config

    def get_context(self) -> SubstitutionList:
        return self.ert_config.substitution_list

    def addDataKW(self, key: str, value: str) -> None:
        self.get_context().addItem(key, value)

    def getObservations(self) -> EnkfObs:
        return self._observations

    def have_observations(self) -> bool:
        return len(self._observations) > 0

    def sample_prior(
        self,
        ensemble: EnsembleAccessor,
        active_realizations: List[int],
        parameters: Optional[List[str]] = None,
    ) -> None:
        """This function is responsible for getting the prior into storage,
        in the case of GEN_KW we sample the data and store it, and if INIT_FILES
        are used without FORWARD_INIT we load files and store them. If FORWARD_INIT
        is set the state is set to INITIALIZED, but no parameters are saved to storage
        until after the forward model has completed.
        """
        t = time.perf_counter()
        parameter_configs = ensemble.experiment.parameter_configuration
        if parameters is None:
            parameters = list(parameter_configs.keys())
        for parameter in parameters:
            config_node = parameter_configs[parameter]
            if config_node.forward_init:
                continue
            for realization_nr in active_realizations:
                ds = config_node.sample_or_load(
                    realization_nr,
                    random_seed=self._global_seed,
                    ensemble_size=ensemble.ensemble_size,
                )
                ensemble.save_parameters(parameter, realization_nr, ds)
        for realization_nr in active_realizations:
            ensemble.update_realization_state(
                realization_nr,
                [
                    RealizationStateEnum.STATE_UNDEFINED,
                    RealizationStateEnum.STATE_LOAD_FAILURE,
                ],
                RealizationStateEnum.STATE_INITIALIZED,
            )

        ensemble.sync()
        logger.debug(f"sample_prior() time_used {(time.perf_counter() - t):.4f}s")

    def rng(self) -> np.random.Generator:
        """Will return the random number generator used for updates."""
        return self._shared_rng

    def createRunPath(self, run_context: RunContext) -> None:
        t = time.perf_counter()
        for iens, run_arg in enumerate(run_context):
            run_path = Path(run_arg.runpath)
            if run_context.is_active(iens):
                run_path.mkdir(parents=True, exist_ok=True)

                for source_file, target_file in self.ert_config.ert_templates:
                    target_file = self.get_context().substitute_real_iter(
                        target_file, run_arg.iens, run_context.iteration
                    )
                    result = self.get_context().substitute_real_iter(
                        Path(source_file).read_text("utf-8"),
                        run_arg.iens,
                        run_context.iteration,
                    )
                    target = run_path / target_file
                    if not target.parent.exists():
                        os.makedirs(
                            target.parent,
                            exist_ok=True,
                        )
                    target.write_text(result)

                ert_config = self.resConfig()
                model_config = ert_config.model_config
                _generate_parameter_files(
                    run_context.sim_fs.experiment.parameter_configuration.values(),
                    model_config.gen_kw_export_name,
                    run_path,
                    run_arg.iens,
                    run_context.sim_fs,
                    run_context.iteration,
                )

                with open(run_path / "jobs.json", mode="w", encoding="utf-8") as fptr:
                    forward_model_output = ert_config.forward_model_data_to_json(
                        run_arg.get_run_id(),
                        run_arg.iens,
                        run_context.iteration,
                    )

                    json.dump(forward_model_output, fptr)

        run_context.runpaths.write_runpath_list(
            [run_context.iteration], run_context.active_realizations
        )

        logger.debug(f"createRunPath() time_used {(time.perf_counter() - t):.4f}s")

    def runWorkflows(
        self,
        runtime: HookRuntime,
        storage: Optional[StorageAccessor] = None,
        ensemble: Optional[EnsembleAccessor] = None,
    ) -> None:
        for workflow in self.ert_config.hooked_workflows[runtime]:
            WorkflowRunner(workflow, self, storage, ensemble).run_blocking()
