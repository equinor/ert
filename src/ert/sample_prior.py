from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from ert.utils import log_duration

from .config import GenKwConfig
from .storage import Ensemble

logger = logging.getLogger(__name__)


@log_duration(
    logger,
)
def sample_prior(
    ensemble: Ensemble,
    active_realizations: Iterable[int],
    random_seed: int,
    parameters: list[str] | None = None,
) -> None:
    """This function is responsible for getting the prior into storage,
    in the case of GEN_KW we sample the data and store it, and if INIT_FILES
    are used without FORWARD_INIT we load files and store them. If FORWARD_INIT
    is set the state is set to INITIALIZED, but no parameters are saved to storage
    until after the forward model has completed.
    """
    parameter_configs = ensemble.experiment.parameter_configuration
    if parameters is None:
        parameters = list(parameter_configs.keys())
    for parameter in parameters:
        config_node = parameter_configs[parameter]
        if config_node.forward_init:
            continue
        logger.info(
            f"Sampling parameter {config_node.name} "
            f"for realizations {active_realizations}"
        )
        if isinstance(config_node, GenKwConfig):
            datasets = [
                Ensemble.sample_parameter(
                    config_node,
                    realization_nr,
                    random_seed=random_seed,
                )
                for realization_nr in active_realizations
            ]
            if datasets:
                ensemble.save_parameters(
                    parameter,
                    realization=None,
                    dataset=pl.concat(datasets, how="vertical"),
                )
        else:
            for realization_nr in active_realizations:
                ds = config_node.read_from_runpath(Path(), realization_nr, 0)
                ensemble.save_parameters(parameter, realization_nr, ds)

    ensemble.refresh_ensemble_state()
