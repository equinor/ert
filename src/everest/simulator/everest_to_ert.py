import itertools
import os
from pathlib import Path
from typing import Any, cast

import everest
from ert.config import (
    ModelConfig,
)
from ert.config.ert_config import _substitutions_from_dict
from ert.config.parsing import ConfigDict
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.plugins import ErtPluginContext
from everest.config import EverestConfig
from everest.config.forward_model_config import SummaryResults
from everest.config.simulator_config import SimulatorConfig
from everest.strings import STORAGE_DIR


def extract_summary_keys(ever_config: EverestConfig) -> list[str]:
    summary_fms = [
        fm
        for fm in ever_config.forward_model
        if fm.results is not None and fm.results.type == "summary"
    ]

    if not summary_fms:
        return []

    summary_fm = summary_fms[0]
    assert summary_fm.results is not None

    smry_results = cast(SummaryResults, summary_fm.results)

    requested_keys: list[str] = ["*"] if smry_results.keys == "*" else smry_results.keys

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS
    deprecated_user_specified_keys = (
        [] if ever_config.export is None else ever_config.export.keywords
    )

    wells = (
        [
            variable.name
            for control in ever_config.controls
            for variable in control.variables
            if control.type == "well_control"
        ]
        if ever_config.wells is None
        else [w.name for w in ever_config.wells]
    )

    well_keys = [
        f"{sum_key}:{wname}"
        for (sum_key, wname) in itertools.product(well_sum_keys, wells)
    ]

    all_keys = data_keys + field_keys + well_keys + deprecated_user_specified_keys

    return list(set(all_keys + requested_keys))


def _extract_environment(
    ever_config: EverestConfig, ert_config: dict[str, Any]
) -> None:
    default_runpath_file = os.path.join(ever_config.output_dir, ".res_runpath_list")
    default_ens_path = os.path.join(ever_config.output_dir, STORAGE_DIR)

    ert_config[ErtConfigKeys.ENSPATH] = default_ens_path
    ert_config[ErtConfigKeys.RUNPATH_FILE] = default_runpath_file


def _extract_simulator(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    """
    Extracts simulation data from ever_config and injects it into ert_config.
    """

    ever_simulation = ever_config.simulator or SimulatorConfig()

    # Resubmit number (number of submission retries)
    ert_config[ErtConfigKeys.MAX_SUBMIT] = ever_simulation.resubmit_limit + 1

    # Maximum number of seconds (MAX_RUNTIME) a forward model is allowed to run
    max_runtime = ever_simulation.max_runtime
    if max_runtime is not None:
        ert_config[ErtConfigKeys.MAX_RUNTIME] = max_runtime or 0

    # Maximum amount of memory (REALIZATION_MEMORY) a forward model is allowed to use
    max_memory = ever_simulation.max_memory
    if max_memory is not None:
        ert_config[ErtConfigKeys.REALIZATION_MEMORY] = max_memory or 0

    # Number of cores reserved on queue nodes (NUM_CPU)
    num_fm_cpu = ever_simulation.cores_per_node
    if num_fm_cpu is not None:
        ert_config[ErtConfigKeys.NUM_CPU] = num_fm_cpu


def _extract_seed(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    random_seed = ever_config.environment.random_seed

    if random_seed:
        ert_config[ErtConfigKeys.RANDOM_SEED] = random_seed


def get_substitutions(
    config_dict: ConfigDict, model_config: ModelConfig, runpath_file: Path, num_cpu: int
) -> dict[str, str]:
    substitutions = _substitutions_from_dict(config_dict)
    substitutions["<RUNPATH_FILE>"] = str(runpath_file)
    substitutions["<RUNPATH>"] = model_config.runpath_format_string
    substitutions["<ECL_BASE>"] = model_config.eclbase_format_string
    substitutions["<ECLBASE>"] = model_config.eclbase_format_string
    substitutions["<NUM_CPU>"] = str(num_cpu)
    return substitutions


def _everest_to_ert_config_dict(ever_config: EverestConfig) -> ConfigDict:
    """
    Takes as input an Everest configuration and converts it
    to a corresponding ert config dict.
    """
    ert_config: dict[str, Any] = {}

    config_dir = ever_config.config_directory
    ert_config[ErtConfigKeys.DEFINE] = [
        ("<CONFIG_PATH>", config_dir),
        ("<CONFIG_FILE>", Path(ever_config.config_file).stem),
    ]

    # Extract simulator and simulation related configs
    _extract_simulator(ever_config, ert_config)
    _extract_environment(ever_config, ert_config)
    _extract_seed(ever_config, ert_config)

    return ert_config


def everest_to_ert_config_dict(everest_config: EverestConfig) -> ConfigDict:
    with ErtPluginContext():
        config_dict = _everest_to_ert_config_dict(everest_config)
    return config_dict
