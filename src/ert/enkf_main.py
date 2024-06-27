from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
from numpy.random import SeedSequence

from .config import ParameterConfig
from .run_context import RunContext
from .runpaths import Runpaths
from .substitution_list import SubstitutionList

if TYPE_CHECKING:
    import numpy.typing as npt

    from .config import ErtConfig
    from .storage import Ensemble

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
    fs: Ensemble,
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
        fs: Ensemble from which to load parameter data
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


def sample_prior(
    ensemble: Ensemble,
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

    logger.debug(f"sample_prior() time_used {(time.perf_counter() - t):.4f}s")


def create_run_path(
    run_context: RunContext,
    ert_config: ErtConfig,
) -> None:
    t = time.perf_counter()
    substitution_list = ert_config.substitution_list
    substitution_list["<ERT-CASE>"] = run_context.ensemble.name
    substitution_list["<ERTCASE>"] = run_context.ensemble.name
    for iens, run_arg in enumerate(run_context):
        run_path = Path(run_arg.runpath)
        if run_context.is_active(iens):
            run_path.mkdir(parents=True, exist_ok=True)

            for source_file, target_file in ert_config.ert_templates:
                target_file = substitution_list.substitute_real_iter(
                    target_file, run_arg.iens, run_context.iteration
                )
                try:
                    file_content = Path(source_file).read_text("utf-8")
                except UnicodeDecodeError as e:
                    raise ValueError(
                        f"Unsupported non UTF-8 character found in file: {source_file}"
                    ) from e

                result = substitution_list.substitute_real_iter(
                    file_content,
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
                run_context.ensemble.experiment.parameter_configuration.values(),
                model_config.gen_kw_export_name,
                run_path,
                run_arg.iens,
                run_context.ensemble,
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
            # Write MANIFEST file to runpath use to avoid NFS sync issues
            with open(run_path / "manifest.json", mode="w", encoding="utf-8") as fptr:
                data = ert_config.manifest_to_json(run_arg.iens, run_arg.itr)
                json.dump(data, fptr)

    run_context.runpaths.write_runpath_list(
        [run_context.iteration], run_context.active_realizations
    )

    logger.debug(f"create_run_path() time_used {(time.perf_counter() - t):.4f}s")


def ensemble_context(
    ensemble: Ensemble,
    active_realizations: npt.NDArray[np.bool_],
    iteration: int,
    substitution_list: Optional[SubstitutionList],
    jobname_format: str,
    runpath_format: str,
    runpath_file: Union[str, Path],
) -> RunContext:
    """This loads an existing ensemble from storage
    and creates run information for that ensemble"""
    substitution_list = (
        SubstitutionList() if substitution_list is None else substitution_list
    )
    run_paths = Runpaths(
        jobname_format=jobname_format,
        runpath_format=runpath_format,
        filename=runpath_file,
        substitution_list=substitution_list,
    )
    return RunContext(
        ensemble=ensemble,
        runpaths=run_paths,
        initial_mask=active_realizations,
        iteration=iteration,
    )
