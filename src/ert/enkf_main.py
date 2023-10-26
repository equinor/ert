from __future__ import annotations

import json
import logging
import os
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import numpy as np
from numpy.random import SeedSequence

from .analysis.configuration import UpdateConfiguration, UpdateStep
from .config import (
    ParameterConfig,
)
from .job_queue import WorkflowRunner
from .realization_state import RealizationState
from .run_context import RunContext
from .runpaths import Runpaths
from .substitution_list import SubstitutionList

if TYPE_CHECKING:
    import numpy.typing as npt

    from .config import ErtConfig, HookRuntime
    from .storage import EnsembleAccessor, EnsembleReader, StorageAccessor

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


def _seed_sequence(seed: Optional[int]) -> int:
    # Set up RNG
    if seed is None:
        int_seed = SeedSequence().entropy
        logger.info(
            "To repeat this experiment, "
            "add the following random seed to your config file:"
        )
        logger.info(f"RANDOM_SEED {int_seed}")
    else:
        int_seed = seed
    assert isinstance(int_seed, int)
    return int_seed


class EnKFMain:
    def __init__(self, config: "ErtConfig", read_only: bool = False) -> None:
        self.ert_config = config
        self._update_configuration: Optional[UpdateConfiguration] = None

    @property
    def update_configuration(self) -> UpdateConfiguration:
        if not self._update_configuration:
            self._update_configuration = UpdateConfiguration.global_update_step(
                list(self.ert_config.observations.keys()),
                self.ert_config.ensemble_config.parameters,
            )
        return self._update_configuration

    @update_configuration.setter
    def update_configuration(self, user_config: List[UpdateStep]) -> None:
        config = UpdateConfiguration(update_steps=user_config)
        config.context_validate(
            list(self.ert_config.observations.keys()),
            self.ert_config.ensemble_config.parameters,
        )
        self._update_configuration = config

    def __repr__(self) -> str:
        return f"EnKFMain(size: {self.ert_config.model_config.num_realizations}, config: {self.ert_config})"

    def runWorkflows(
        self,
        runtime: HookRuntime,
        storage: Optional[StorageAccessor] = None,
        ensemble: Optional[EnsembleAccessor] = None,
    ) -> None:
        for workflow in self.ert_config.hooked_workflows[runtime]:
            WorkflowRunner(workflow, self, storage, ensemble).run_blocking()


def sample_prior(
    ensemble: EnsembleAccessor,
    active_realizations: Iterable[int],
    parameters: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
) -> None:
    """This function is responsible for getting the prior into storage,
    in the case of GEN_KW we sample the data and store it, and if INIT_FILES
    are used without FORWARD_INIT we load files and store them. If FORWARD_INIT
    is set the state is set to INITIALIZED, but no parameters are saved to storage
    until after the forward model has completed.
    """
    random_seed = _seed_sequence(random_seed)
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
                random_seed=random_seed,
                ensemble_size=ensemble.ensemble_size,
            )
            ensemble.save_parameters(parameter, realization_nr, ds)
    for realization_nr in active_realizations:
        ensemble.update_realization_state(
            realization_nr,
            [
                RealizationState.UNDEFINED,
                RealizationState.LOAD_FAILURE,
            ],
            RealizationState.INITIALIZED,
        )

    ensemble.sync()
    logger.debug(f"sample_prior() time_used {(time.perf_counter() - t):.4f}s")


def create_run_path(
    run_context: RunContext,
    substitution_list: SubstitutionList,
    ert_config: ErtConfig,
) -> None:
    t = time.perf_counter()
    substitution_list = copy(substitution_list)
    substitution_list["<ERT-CASE>"] = run_context.sim_fs.name
    substitution_list["<ERTCASE>"] = run_context.sim_fs.name
    for iens, run_arg in enumerate(run_context):
        run_path = Path(run_arg.runpath)
        if run_context.is_active(iens):
            run_path.mkdir(parents=True, exist_ok=True)

            for source_file, target_file in ert_config.ert_templates:
                target_file = substitution_list.substitute_real_iter(
                    target_file, run_arg.iens, run_context.iteration
                )
                result = substitution_list.substitute_real_iter(
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

            model_config = ert_config.model_config
            _generate_parameter_files(
                run_context.sim_fs.experiment.parameter_configuration.values(),
                model_config.gen_kw_export_name,
                run_path,
                run_arg.iens,
                run_context.sim_fs,
                run_context.iteration,
            )

            path = run_path / "jobs.json"
            _backup_if_existing(path)
            with open(run_path / "jobs.json", mode="w", encoding="utf-8") as fptr:
                forward_model_output = ert_config.forward_model_data_to_json(
                    run_arg.run_id,
                    run_arg.iens,
                    run_context.iteration,
                )

                json.dump(forward_model_output, fptr)

    run_context.runpaths.write_runpath_list(
        [run_context.iteration], run_context.active_realizations
    )

    logger.debug(f"create_run_path() time_used {(time.perf_counter() - t):.4f}s")


def ensemble_context(
    case: EnsembleAccessor,
    active_realizations: npt.NDArray[np.bool_],
    iteration: int,
    substitution_list: Optional[SubstitutionList],
    jobname_format: str,
    runpath_format: str,
    runpath_file: Union[str, Path],
) -> RunContext:
    """This loads an existing case from storage
    and creates run information for that case"""
    substitution_list = (
        SubstitutionList() if substitution_list is None else substitution_list
    )
    run_paths = Runpaths(
        jobname_format=jobname_format,
        runpath_format=runpath_format,
        filename=runpath_file,
        substitute=substitution_list.substitute_real_iter,
    )
    return RunContext(
        sim_fs=case,
        runpaths=run_paths,
        initial_mask=active_realizations,
        iteration=iteration,
    )
