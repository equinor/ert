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
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import orjson
from numpy.random import SeedSequence

from .config import (
    ExtParamConfig,
    Field,
    GenKwConfig,
    ParameterConfig,
    SurfaceConfig,
)
from .run_arg import RunArg
from .runpaths import Runpaths

if TYPE_CHECKING:
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
    runarg: RunArg,
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
        export_values = node.write_to_runpath(runarg.file_in_runpath, runarg.iens, fs)
        if export_values:
            exports.update(export_values)
        continue

    _value_export_txt(Path(runarg.runpath), export_base_name, exports)
    _value_export_json(Path(runarg.runpath), export_base_name, exports)


def _manifest_to_json(
    ensemble: Ensemble, file_in_runpath: Callable[[str], str], iens: int
) -> Dict[str, Any]:
    manifest = {}
    # Add expected parameter files to manifest
    for param_config in ensemble.experiment.parameter_configuration.values():
        assert isinstance(
            param_config,
            (ExtParamConfig, GenKwConfig, Field, SurfaceConfig),
        )
        if param_config.forward_init and ensemble.iteration == 0:
            assert param_config.forward_init_file is not None
            file_path = param_config.forward_init_file.replace("%d", str(iens))
            manifest[param_config.name] = file_in_runpath(file_path)
    # Add expected response files to manifest
    for respons_config in ensemble.experiment.response_configuration.values():
        for input_file in respons_config.expected_input_files:
            manifest[f"{respons_config.response_type}_{input_file}"] = file_in_runpath(
                input_file
            )

    return manifest


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

    def file_in_config_path(real: int) -> Callable[[str], str]:
        def inner(filename: str) -> str:
            if "%d" in filename:
                filename = filename % real  # noqa
            return filename.replace("<IENS>", str(real)).replace("<ITER>", "0")

        return inner

    for parameter in parameters:
        config_node = parameter_configs[parameter]
        if config_node.forward_init:
            continue
        for realization_nr in active_realizations:
            ds = config_node.sample_or_load(
                file_in_config_path(realization_nr),
                realization_nr,
                random_seed=random_seed,
                ensemble_size=ensemble.ensemble_size,
            )
            ensemble.save_parameters(parameter, realization_nr, ds)

    logger.debug(f"sample_prior() time_used {(time.perf_counter() - t):.4f}s")


def create_run_path(
    run_args: List[RunArg],
    ensemble: Ensemble,
    ert_config: ErtConfig,
    runpaths: Runpaths,
    context_env: Optional[Dict[str, str]] = None,
) -> None:
    if context_env is None:
        context_env = {}
    t = time.perf_counter()
    substitutions = ert_config.substitutions
    runpaths.set_ert_ensemble(ensemble.name)
    for run_arg in run_args:
        run_path = Path(run_arg.runpath)
        if run_arg.active:
            run_path.mkdir(parents=True, exist_ok=True)
            for source_file, target_file in ert_config.ert_templates:
                target_file = substitutions.substitute_real_iter(
                    target_file, run_arg.iens, ensemble.iteration
                )
                try:
                    file_content = Path(source_file).read_text("utf-8")
                except UnicodeDecodeError as e:
                    raise ValueError(
                        f"Unsupported non UTF-8 character found in file: {source_file}"
                    ) from e

                result = substitutions.substitute_real_iter(
                    file_content,
                    run_arg.iens,
                    ensemble.iteration,
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
                ensemble.experiment.parameter_configuration.values(),
                model_config.gen_kw_export_name,
                run_arg,
                ensemble,
                ensemble.iteration,
            )

            path = run_path / "jobs.json"
            _backup_if_existing(path)
            forward_model_output = ert_config.forward_model_data_to_json(
                run_arg.run_id, run_arg.iens, ensemble.iteration, context_env
            )
            with open(run_path / "jobs.json", mode="wb") as fptr:
                fptr.write(
                    orjson.dumps(forward_model_output, option=orjson.OPT_NON_STR_KEYS)
                )
            # Write MANIFEST file to runpath use to avoid NFS sync issues
            data = _manifest_to_json(ensemble, run_arg.file_in_runpath, run_arg.iens)
            with open(run_path / "manifest.json", mode="wb") as fptr:
                fptr.write(orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS))

    runpaths.write_runpath_list(
        [ensemble.iteration], [real.iens for real in run_args if real.active]
    )

    logger.debug(f"create_run_path() time_used {(time.perf_counter() - t):.4f}s")
