from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

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
                logger.info(
                    f"Getting parameter {config_node.name} "
                    f"from design matrix for realizations {active_realizations}"
                )
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
                logger.info(
                    f"Sampling parameter {config_node.name} "
                    f"for realizations {active_realizations}"
                )
                datasets = [
                    Ensemble.sample_parameter(
                        config_node,
                        realization_nr,
                        random_seed=random_seed,
                    )
                    for realization_nr in active_realizations
                ]
                if datasets:
                    dataset = pl.concat(datasets, how="vertical")

            if dataset is not None:
                ensemble.save_parameters(
                    dataset=dataset,
                )
        else:
            for realization_nr in active_realizations:
                ds = config_node.read_from_runpath(Path(), realization_nr, 0)
                ensemble.save_parameters(ds, parameter, realization_nr)

    ensemble.refresh_ensemble_state()
