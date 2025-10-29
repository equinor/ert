import itertools
import warnings
from typing import Any, cast

import everest
from ert.base_model_context import use_runtime_plugins
from ert.config.parsing import ConfigDict
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.plugins import get_site_plugins
from everest.config import EverestConfig
from everest.config.forward_model_config import SummaryResults
from everest.config.simulator_config import SimulatorConfig


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
    if (num_fm_cpu := ever_simulation.cores_per_node) is not None:
        if (
            ever_simulation.queue_system is not None
            and "num_cpu" not in ever_simulation.queue_system.model_fields_set
        ):
            ert_config[ErtConfigKeys.NUM_CPU] = num_fm_cpu
        else:
            warnings.warn(
                "Ignoring cores_per_node as num_cpu was set", UserWarning, stacklevel=2
            )


def _everest_to_ert_config_dict(ever_config: EverestConfig) -> ConfigDict:
    """
    Takes as input an Everest configuration and converts it
    to a corresponding ert config dict.
    """
    ert_config: dict[str, Any] = {}

    # Extract simulator and simulation related configs
    _extract_simulator(ever_config, ert_config)

    return ert_config


def everest_to_ert_config_dict(everest_config: EverestConfig) -> ConfigDict:
    with use_runtime_plugins(get_site_plugins()):
        config_dict = _everest_to_ert_config_dict(everest_config)
    return config_dict
