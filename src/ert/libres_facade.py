from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from pandas import DataFrame

from ert.analysis import AnalysisEvent, SmootherSnapshot, smoother_update
from ert.config import (
    ErtConfig,
    Field,
    ObservationType,
)
from ert.data import MeasuredData
from ert.data._measured_data import ObservationError, ResponseError

from .plugins import ErtPluginContext

if TYPE_CHECKING:
    from ert.config import (
        EnkfObs,
    )
    from ert.storage import Ensemble


class LibresFacade:
    """The intention of this class is to expose properties or data of ert
    commonly used in other project. It is part of the public interface of ert,
    and as such changes here should not be taken lightly."""

    def __init__(self, ert_config: ErtConfig, _: Any = None) -> None:
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
    ) -> SmootherSnapshot:
        if rng is None:
            rng = np.random.default_rng()
        update_snapshot = smoother_update(
            prior_storage,
            posterior_storage,
            observations,
            parameters,
            self.config.analysis_config.observation_settings,
            self.config.analysis_config.es_settings,
            rng,
            progress_callback,
            global_std_scaling,
        )
        self.update_snapshots[run_id] = update_snapshot
        return update_snapshot

    @property
    def user_config_file(self) -> str | None:
        return self.config.user_config_file

    def get_field_parameters(self) -> list[str]:
        return [
            val.name
            for val in self.config.ensemble_config.parameter_configuration
            if isinstance(val, Field)
        ]

    @property
    def run_path(self) -> str:
        return self.config.runpath_config.runpath_format_string

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

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> LibresFacade:
        with ErtPluginContext():
            return cls(
                ErtConfig.with_plugins().from_file(config_file),
                read_only,
            )
