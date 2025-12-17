from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from _ert.utils import file_safe_timestamp
from ert.config import (
    ExtParamConfig,
    Field,
    ForwardModelStep,
    GenKwConfig,
    ParameterCardinality,
    ParameterConfig,
    SurfaceConfig,
)
from ert.config.design_matrix import DESIGN_MATRIX_GROUP
from ert.config.distribution import LogNormalSettings, LogUnifSettings
from ert.config.ert_config import create_forward_model_json
from ert.substitutions import Substitutions, substitute_runpath_name
from ert.utils import log_duration

if TYPE_CHECKING:
    from ert.run_arg import RunArg
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


def _conditionally_format_float(num: float) -> str:
    str_num = str(num)
    if "." not in str_num:
        return str_num
    float_parts = str_num.split(".")
    formatted_str = f"{num:.6f}" if len(float_parts[1]) >= 6 else str(num)
    return formatted_str.rstrip("0").rstrip(".")


def _backup_if_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = file_safe_timestamp(datetime.now(UTC).isoformat(timespec="seconds"))
    new_path = path.parent / f"{path.name}_backup_{timestamp}"
    path.rename(new_path)


def _value_export_txt(
    run_path: Path,
    export_base_name: str,
    values: Mapping[str, Mapping[str, float | str]],
) -> None:
    path = run_path / f"{export_base_name}.txt"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    with path.open("w") as f:
        for key, param_map in values.items():
            for param, value in param_map.items():
                if isinstance(value, (int)):
                    if key == DESIGN_MATRIX_GROUP:
                        print(f"{param} {value}", file=f)
                    else:
                        print(f"{key}:{param} {value}", file=f)
                elif isinstance(value, (float)):
                    if key == DESIGN_MATRIX_GROUP:
                        print(f"{param} {_conditionally_format_float(value)}", file=f)
                    else:
                        print(
                            f"{key}:{param} {_conditionally_format_float(value)}",
                            file=f,
                        )
                elif key == DESIGN_MATRIX_GROUP:
                    print(f"{param} {value}", file=f)
                else:
                    print(f"{key}:{param} {value}", file=f)


def _value_export_json(
    run_path: Path,
    export_base_name: str,
    values: Mapping[str, Mapping[str, float | str]],
) -> None:
    path = run_path / f"{export_base_name}.json"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    # parameter file is {param: {"value": value}}
    json_out: dict[str, dict[str, float | str]] = {}
    for param_map in values.values():
        for param, value in param_map.items():
            json_out[param] = {"value": value}

    # Disallow NaN from being written: ERT produces the parameters and the only
    # way for the output to be NaN is if the input is invalid or if the sampling
    # function is buggy. Either way, that would be a bug and we can report it by
    # having json throw an error.
    with path.open("w") as f:
        json.dump(json_out, f, allow_nan=False, indent=0, separators=(", ", " : "))


def _generate_parameter_files(
    parameter_configs: Iterable[ParameterConfig],
    export_base_name: str,
    run_path: Path,
    iens: int,
    fs: Ensemble,
    iteration: int,
) -> Mapping[str, Mapping[str, float | str]]:
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

    Returns:
        Returns the union of parameters returned by write_to_runpath for each
        parameter_config.
    """
    # preload scalar parameters for this realization
    keys = [
        p.name
        for p in parameter_configs
        if p.cardinality == ParameterCardinality.multiple_configs_per_ensemble_dataset
    ]
    scalar_data: dict[str, float | str] = {}
    if keys:
        df = fs._load_scalar_keys(keys=keys, realizations=iens, transformed=True)
        scalar_data = df.to_dicts()[0]
    exports: dict[str, dict[str, float | str]] = {}
    log_exports: dict[str, dict[str, float | str]] = {}
    for param in parameter_configs:
        # For the first iteration we do not write the parameter
        # to run path, as we expect to read if after the forward
        # model has completed.
        if param.forward_init and iteration == 0:
            continue
        export_values: dict[str, dict[str, float | str]] | None = None
        log_export_values: dict[str, dict[str, float | str]] | None = {}
        if param.name in scalar_data:
            scalar_value = scalar_data[param.name]
            export_values = {param.group_name: {param.name: scalar_value}}
            if isinstance(param, GenKwConfig) and isinstance(
                param.distribution, (LogNormalSettings, LogUnifSettings)
            ):
                if isinstance(scalar_value, float) and scalar_value > 0:
                    log_value = math.log10(scalar_value)
                    log_export_values = {
                        f"LOG10_{param.group_name}": {param.name: log_value}
                    }
                else:
                    logger.warning(
                        "Could not export the log10 value of "
                        f"{scalar_value} as it is invalid"
                    )
        else:
            export_values = param.write_to_runpath(Path(run_path), iens, fs)
        if export_values:
            for group, vals in export_values.items():
                exports.setdefault(group, {}).update(vals)
        if log_export_values:
            for group, vals in log_export_values.items():
                log_exports.setdefault(group, {}).update(vals)
        continue

    _value_export_txt(run_path, export_base_name, exports | log_exports)
    _value_export_json(run_path, export_base_name, exports)
    return exports


def _manifest_to_json(ensemble: Ensemble, iens: int, iter_: int) -> dict[str, Any]:
    manifest = {}
    # Add expected parameter files to manifest
    for param_config in ensemble.experiment.parameter_configuration.values():
        assert isinstance(
            param_config,
            ExtParamConfig | GenKwConfig | Field | SurfaceConfig,
        )
        if param_config.forward_init and ensemble.iteration == 0:
            assert not isinstance(param_config, GenKwConfig)
            assert param_config.forward_init_file is not None
            file_path = substitute_runpath_name(
                param_config.forward_init_file, iens, iter_
            )
            manifest[param_config.name] = file_path
    # Add expected response files to manifest
    for response_config in ensemble.experiment.response_configuration.values():
        for input_file in response_config.expected_input_files:
            manifest[f"{response_config.type}_{input_file}"] = substitute_runpath_name(
                input_file, iens, iter_
            )

    return manifest


def _make_param_substituter(
    substituter: Substitutions,
    param_data: Mapping[str, Mapping[str, str | float]],
) -> Substitutions:
    param_substituter = deepcopy(substituter)
    for values in param_data.values():
        for param_name, value in values.items():
            formatted_value = (
                _conditionally_format_float(value)
                if isinstance(value, float)
                else str(value)
            )
            param_substituter[f"<{param_name}>"] = formatted_value
    return param_substituter


@log_duration(logger, logging.INFO)
def create_run_path(
    run_args: list[RunArg],
    ensemble: Ensemble,
    user_config_file: str,
    env_vars: dict[str, str],
    env_pr_fm_step: dict[str, dict[str, Any]],
    forward_model_steps: list[ForwardModelStep],
    substitutions: dict[str, str],
    parameters_file: str,
    runpaths: Runpaths,
    context_env: dict[str, str] | None = None,
) -> None:
    if context_env is None:
        context_env = {}
    runpaths.set_ert_ensemble(ensemble.name)
    timings = {
        "generate_parameter_files": 0.0,
        "substitute_parameters": 0.0,
        "substitute_real_iter": 0.0,
        "result_file_to_target": 0.0,
    }
    substituter = Substitutions(substitutions)
    for run_arg in run_args:
        run_path = Path(run_arg.runpath)
        if run_arg.active:
            run_path.mkdir(parents=True, exist_ok=True)
            start_time = time.perf_counter()
            param_data = _generate_parameter_files(
                ensemble.experiment.parameter_configuration.values(),
                parameters_file,
                run_path,
                run_arg.iens,
                ensemble,
                ensemble.iteration,
            )
            timings["generate_parameter_files"] += time.perf_counter() - start_time
            real_iter_substituter = substituter.real_iter_substituter(
                run_arg.iens, ensemble.iteration
            )
            param_substituter = _make_param_substituter(
                real_iter_substituter, param_data
            )
            for (
                source_file_content,
                target_file,
            ) in ensemble.experiment.templates_configuration:
                start_time = time.perf_counter()
                target_file = real_iter_substituter.substitute(target_file)
                timings["substitute_real_iter"] += time.perf_counter() - start_time
                start_time = time.perf_counter()
                result = param_substituter.substitute(source_file_content)
                timings["substitute_parameters"] += time.perf_counter() - start_time
                start_time = time.perf_counter()
                target = run_path / target_file
                if not target.parent.exists():
                    os.makedirs(
                        target.parent,
                        exist_ok=True,
                    )
                target.write_text(result)
                timings["result_file_to_target"] += time.perf_counter() - start_time

            path = run_path / "jobs.json"
            start_time = time.perf_counter()
            _backup_if_existing(path)
            timings["backup_if_existing"] = time.perf_counter() - start_time
            start_time = time.perf_counter()
            forward_model_output = create_forward_model_json(
                context=substitutions,
                forward_model_steps=forward_model_steps,
                user_config_file=user_config_file,
                env_vars={**env_vars, **context_env},
                env_pr_fm_step=env_pr_fm_step,
                run_id=run_arg.run_id,
                iens=run_arg.iens,
                itr=ensemble.iteration,
            )
            Path(run_path / "jobs.json").write_bytes(
                orjson.dumps(
                    forward_model_output,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
                )
            )
            timings["jobs_to_json"] = time.perf_counter() - start_time
            # Write MANIFEST file to runpath use to avoid NFS sync issues
            start_time = time.perf_counter()
            data = _manifest_to_json(ensemble, run_arg.iens, run_arg.itr)
            Path(run_path / "manifest.json").write_bytes(
                orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2)
            )
            timings["manifest_to_json"] = time.perf_counter() - start_time
    logger.info(f"_create_run_path durations: {timings}")
    runpaths.write_runpath_list(
        [ensemble.iteration], [real.iens for real in run_args if real.active]
    )
