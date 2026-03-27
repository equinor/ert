from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from ert.config.parameter_config import ParameterConfig
from ert.utils import log_duration

from .config import DataSource, GenKwConfig
from .storage import Ensemble

logger = logging.getLogger(__name__)


@log_duration(
    logger,
)
def sample_prior(
    ensemble: Ensemble,
    active_realizations: Iterable[int],
    random_seed: int,
    num_realizations: int,
    parameters: list[str] | None = None,
    design_matrix_df: pl.DataFrame | None = None,
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
    complete_dataset: pl.DataFrame | None = None

    log_params(parameter_configs, design_matrix_df, active_realizations)

    for parameter in parameters:
        config_node = parameter_configs[parameter]
        if config_node.forward_init:
            continue

        if isinstance(config_node, GenKwConfig):
            dataset: pl.DataFrame | None = None
            if (
                config_node.input_source == DataSource.DESIGN_MATRIX
                and design_matrix_df is not None
            ):
                cols = {"realization", config_node.name}
                missing = cols - set(design_matrix_df.columns)
                if missing:
                    raise KeyError(
                        f"Design matrix is missing column(s): {', '.join(missing)}"
                    )
                dataset = design_matrix_df.select(
                    ["realization", config_node.name]
                ).filter(pl.col("realization").is_in(list(active_realizations)))
                if dataset.is_empty():
                    raise KeyError("Active realization mask is not in design matrix!")
            elif config_node.input_source == DataSource.SAMPLED:
                dataset = Ensemble.sample_parameter(
                    config_node,
                    list(active_realizations),
                    random_seed=random_seed,
                    num_realizations=num_realizations,
                )
            if not (dataset is None or dataset.is_empty()):
                if complete_dataset is None:
                    complete_dataset = dataset
                elif dataset is not None:
                    complete_dataset = complete_dataset.join(dataset, on="realization")

        else:
            for realization_nr in active_realizations:
                ds = config_node.read_from_runpath(Path(), realization_nr, 0)
                ensemble.save_parameters(ds, parameter, realization_nr)

    if complete_dataset is not None:
        ensemble.save_parameters(
            dataset=complete_dataset,
        )
    ensemble.refresh_ensemble_state()


def log_params(
    param_configs: dict[str, ParameterConfig],
    design_matrix_df: pl.DataFrame | None,
    active_realizations: Iterable[int],
) -> None:
    sample_params: list[str] = [
        p.name
        for p in param_configs.values()
        if (isinstance(p, GenKwConfig) and p.input_source == DataSource.SAMPLED)
    ]
    get_params: list[str] = [
        p.name
        for p in param_configs.values()
        if (
            isinstance(p, GenKwConfig)
            and (
                p.input_source == DataSource.DESIGN_MATRIX
                and design_matrix_df is not None
            )
        )
    ]

    if len(sample_params) > 0:
        logger.info(
            f"Sampling parameters: {', '.join(sample_params)}"
            f" for realizations {active_realizations}"
        )

    if len(get_params) > 0:
        logger.info(
            f"Getting parameters: {', '.join(get_params)}"
            f" from design matrix for realizations {active_realizations}"
        )
