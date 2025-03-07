from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Callable, Iterable
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from pandas import DataFrame

from ert.analysis import AnalysisEvent, SmootherSnapshot, smoother_update
from ert.callbacks import forward_model_ok
from ert.config import (
    ErtConfig,
    Field,
    ObservationType,
)
from ert.data import MeasuredData
from ert.data._measured_data import ObservationError, ResponseError
from ert.load_status import LoadResult, LoadStatus

from .plugins import ErtPluginContext

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ert.config import (
        EnkfObs,
    )
    from ert.storage import Ensemble, Storage


def _load_realization_from_run_path(
    run_path: str,
    realization: int,
    ensemble: Ensemble,
) -> tuple[LoadResult, int]:
    result = asyncio.run(forward_model_ok(run_path, realization, 0, ensemble))
    return result, realization


class LibresFacade:
    """The intention of this class is to expose properties or data of ert
    commonly used in other project. It is part of the public interface of ert,
    and as such changes here should not be taken lightly."""

    def __init__(self, ert_config: ErtConfig, _: Any = None):
        self.config = ert_config
        self.update_snapshots: dict[str, SmootherSnapshot] = {}
        self.update_configuration = None

    def smoother_update(
        self,
        prior_storage: Ensemble,
        posterior_storage: Ensemble,
        run_id: str,
        observations: Iterable[str],
        parameters: Iterable[str],
        progress_callback: Callable[[AnalysisEvent], None] | None = None,
        global_std_scaling: float = 1.0,
        rng: np.random.Generator | None = None,
        misfit_process: bool = False,
    ) -> SmootherSnapshot:
        if rng is None:
            rng = np.random.default_rng()
        update_snapshot = smoother_update(
            prior_storage,
            posterior_storage,
            observations,
            parameters,
            self.config.analysis_config.observation_settings,
            self.config.analysis_config.es_module,
            rng,
            progress_callback,
            global_std_scaling,
        )
        self.update_snapshots[run_id] = update_snapshot
        return update_snapshot

    @property
    def enspath(self) -> str:
        return self.config.ens_path

    @property
    def user_config_file(self) -> str | None:
        return self.config.user_config_file

    def get_field_parameters(self) -> list[str]:
        return [
            val.name
            for val in self.config.ensemble_config.parameter_configuration
            if isinstance(val, Field)
        ]

    def get_ensemble_size(self) -> int:
        return self.config.model_config.num_realizations

    @property
    def run_path(self) -> str:
        return self.config.model_config.runpath_format_string

    @property
    def resolved_run_path(self) -> str:
        return str(Path(self.config.model_config.runpath_format_string).resolve())

    @staticmethod
    def load_from_run_path(
        run_path_format: str,
        ensemble: Ensemble,
        active_realizations: list[int],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization_from_run_path,
                (
                    run_path_format.replace("<IENS>", str(realization)).replace(
                        "<ITER>", "0"
                    ),
                    realization,
                    ensemble,
                ),
            )
            for realization in active_realizations
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
            else:
                _logger.error(f"Realization: {iens}, load failure: {message}")

        ensemble.refresh_ensemble_state()
        return loaded

    def get_observations(self) -> EnkfObs:
        return self.config.enkf_obs

    def get_data_key_for_obs_key(self, observation_key: str) -> str:
        obs = self.config.enkf_obs[observation_key]
        if obs.observation_type == ObservationType.SUMMARY:
            return next(iter(obs.observations.values())).summary_key  # type: ignore
        else:
            return obs.data_key

    @staticmethod
    def load_all_misfit_data(ensemble: Ensemble) -> DataFrame:
        """Loads all misfit data for a given ensemble.

        Retrieves all active realizations from the ensemble, and for each
        realization, it gathers the observations and measured data. The
        function then calculates the misfit, which is a measure of the
        discrepancy between observed and simulated values, for each data
        column. The misfit is calculated as the squared difference between the
        observed and measured data, normalized by the standard deviation of the
        observations.

        The misfit data is then grouped by key, summed, and transposed to form
        a DataFrame. The DataFrame has an additional column "MISFIT:TOTAL",
        which is the sum of all misfits for each realization. The index of the
        DataFrame is named "Realization".

        Parameters:
            ensemble: The ensemble from which to load the misfit data.

        Returns:
            DataFrame: A DataFrame containing the misfit data for all
                realizations in the ensemble. Each column (except for "MISFIT:TOTAL")
                corresponds to a key in the measured data, and each row corresponds
                to a realization. The "MISFIT:TOTAL" column contains the total
                misfit for each realization.
        """
        try:
            measured_data = MeasuredData(ensemble)
        except (ResponseError, ObservationError):
            return DataFrame()
        misfit = DataFrame()
        for name in measured_data.data.columns.unique(0):
            df = (
                (
                    measured_data.data[name].loc["OBS"]
                    - measured_data.get_simulated_data()[name]
                )
                / measured_data.data[name].loc["STD"]
            ) ** 2
            misfit[f"MISFIT:{name}"] = df.sum(axis=1)
        misfit["MISFIT:TOTAL"] = misfit.sum(axis=1)
        misfit.index.name = "Realization"
        misfit.index = misfit.index.astype(int)

        return misfit

    def run_ertscript(  # type: ignore
        self,
        ertscript,
        storage: Storage,
        ensemble: Ensemble,
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> Any:
        warnings.warn(
            "run_ertscript is deprecated, use the workflow runner",
            DeprecationWarning,
            stacklevel=1,
        )
        return ertscript().initializeAndRun(
            [],
            argument_values=args,
            fixtures={
                "storage": storage,
                "ensemble": ensemble,
                "reports_dir": (
                    self.config.analysis_config.log_path / ensemble.experiment.name
                ),
                "observation_settings": self.config.analysis_config.observation_settings,
                "es_settings": self.config.analysis_config.es_module,
                "random_seed": self.config.random_seed,
            },
            **kwargs,
        )

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> LibresFacade:
        with ErtPluginContext():
            return cls(
                ErtConfig.with_plugins().from_file(config_file),
                read_only,
            )
