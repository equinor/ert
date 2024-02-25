from __future__ import annotations

import logging
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from deprecation import deprecated
from pandas import DataFrame
from resdata.grid import Grid

from ert.analysis import AnalysisEvent, SmootherSnapshot, smoother_update
from ert.callbacks import forward_model_ok
from ert.config import (
    EnkfObservationImplementationType,
    ErtConfig,
    Field,
    GenKwConfig,
)
from ert.data import MeasuredData
from ert.data._measured_data import ObservationError, ResponseError
from ert.load_status import LoadResult, LoadStatus
from ert.shared.version import __version__
from ert.storage import Ensemble

from .enkf_main import EnKFMain, ensemble_context

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import (
        EnkfObs,
        PriorDict,
        WorkflowJob,
    )
    from ert.run_arg import RunArg
    from ert.storage import Ensemble, Storage


def _load_realization(
    realisation: int,
    run_args: List[RunArg],
) -> Tuple[LoadResult, int]:
    result = forward_model_ok(run_args[realisation])
    return result, realisation


class LibresFacade:
    """The intention of this class is to expose properties or data of ert
    commonly used in other project. It is part of the public interface of ert,
    and as such changes here should not be taken lightly."""

    def __init__(self, enkf_main: Union[EnKFMain, ErtConfig]):
        # EnKFMain is more or less just a facade for the configuration at this
        # point, so in the process of removing it altogether it is easier
        # if we allow the facade to created with both EnKFMain and ErtConfig
        if isinstance(enkf_main, EnKFMain):
            self._enkf_main = enkf_main
            self.config: ErtConfig = enkf_main.ert_config
        else:
            self._enkf_main = EnKFMain(enkf_main)
            self.config = enkf_main
        self.update_snapshots: Dict[str, SmootherSnapshot] = {}
        self.update_configuration = None

    def smoother_update(
        self,
        prior_storage: Ensemble,
        posterior_storage: Ensemble,
        run_id: str,
        observations: Iterable[str],
        parameters: Iterable[str],
        progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
        global_std_scaling: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        misfit_process: bool = False,
    ) -> SmootherSnapshot:
        if rng is None:
            rng = np.random.default_rng()
        update_snapshot = smoother_update(
            prior_storage,
            posterior_storage,
            run_id,
            observations,
            parameters,
            self.config.analysis_config.observation_settings,
            self.config.analysis_config.es_module,
            rng,
            progress_callback,
            global_std_scaling,
            log_path=self.config.analysis_config.log_path,
        )
        self.update_snapshots[run_id] = update_snapshot
        return update_snapshot

    @property
    def enspath(self) -> str:
        return self.config.ens_path

    @property
    def user_config_file(self) -> Optional[str]:
        return self.config.user_config_file

    def get_field_parameters(self) -> List[str]:
        return list(
            val.name
            for val in self.config.ensemble_config.parameter_configuration
            if isinstance(val, Field)
        )

    def get_gen_kw(self) -> List[str]:
        return self.config.ensemble_config.get_keylist_gen_kw()

    @property
    def grid_file(self) -> Optional[str]:
        return self.config.ensemble_config.grid_file

    @property
    @deprecated(
        deprecated_in="5.0",
        current_version=__version__,
        details=(
            "Grid's statefullness is linked to the configuration and not stored data.\n"
            "This can lead to inconsistencies between field and grid."
        ),
    )  # type: ignore
    def grid(self) -> Optional[Grid]:
        path = self.grid_file
        if path:
            ext = Path(path).suffix
            if ext in [".EGRID", ".GRID"]:
                return Grid.load_from_file(path)
        return None

    def get_ensemble_size(self) -> int:
        return self.config.model_config.num_realizations

    @property
    def run_path(self) -> str:
        return self.config.model_config.runpath_format_string

    def load_from_forward_model(
        self,
        ensemble: Ensemble,
        realisations: npt.NDArray[np.bool_],
        iteration: int,
    ) -> int:
        t = time.perf_counter()
        run_context = ensemble_context(
            ensemble,
            realisations,
            iteration,
            self.config.substitution_list,
            jobname_format=self.config.model_config.jobname_format_string,
            runpath_format=self.config.model_config.runpath_format_string,
            runpath_file=self.config.runpath_file,
        )

        nr_loaded = self._load_from_run_path(
            self.config.model_config.num_realizations,
            run_context.run_args,
            run_context.mask,
        )
        _logger.debug(
            f"load_from_forward_model() time_used {(time.perf_counter() - t):.4f}s"
        )
        return nr_loaded

    def _load_from_run_path(
        self,
        ensemble_size: int,
        run_args: List[RunArg],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization,
                (iens, run_args),
            )
            for iens in range(ensemble_size)
            if active_realizations[iens]
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
            else:
                _logger.error(f"Realization: {iens}, load failure: {message}")

        return loaded

    def get_observations(self) -> "EnkfObs":
        return self.config.enkf_obs

    def get_data_key_for_obs_key(self, observation_key: str) -> str:
        obs = self.config.enkf_obs[observation_key]
        if obs.observation_type == EnkfObservationImplementationType.SUMMARY_OBS:
            return list(obs.observations.values())[0].summary_key  # type: ignore
        else:
            return obs.data_key

    def get_gen_data_keys(self) -> List[str]:
        ensemble_config = self.config.ensemble_config
        gen_data_keys = ensemble_config.get_keylist_gen_data()
        gen_data_list = []
        for key in gen_data_keys:
            gen_data_config = ensemble_config.getNodeGenData(key)
            if gen_data_config.report_steps is None:
                gen_data_list.append(f"{key}@0")
            else:
                for report_step in gen_data_config.report_steps:
                    gen_data_list.append(f"{key}@{report_step}")
        return sorted(gen_data_list, key=lambda k: k.lower())

    def all_data_type_keys(self) -> List[str]:
        return self.get_summary_keys() + self.gen_kw_keys() + self.get_gen_data_keys()

    def observation_keys(self, key: str) -> List[str]:
        if key in self.get_gen_data_keys():
            key_parts = key.split("@")
            data_key = key_parts[0]
            data_report_step = int(key_parts[1]) if len(key_parts) > 1 else 0

            obs_key = None

            enkf_obs = self.config.enkf_obs
            for obs_vector in enkf_obs:
                if EnkfObservationImplementationType.GEN_OBS:
                    report_step = min(obs_vector.observations.keys())
                    key = obs_vector.data_key

                    if key == data_key and report_step == data_report_step:
                        obs_key = obs_vector.observation_key
            if obs_key is not None:
                return [obs_key]
            else:
                return []
        elif key in self.get_summary_keys():
            obs = self.get_observations().getTypedKeylist(
                EnkfObservationImplementationType.SUMMARY_OBS
            )
            return [i for i in obs if self.get_observations()[i].observation_key == key]
        else:
            return []

    def load_all_misfit_data(self, ensemble: Ensemble) -> DataFrame:
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

    def get_summary_keys(self) -> List[str]:
        return self.config.ensemble_config.get_summary_keys()

    def gen_kw_keys(self) -> List[str]:
        gen_kw_keys = self.get_gen_kw()

        gen_kw_list = []
        for key in gen_kw_keys:
            gen_kw_config = self.config.ensemble_config.parameter_configs[key]
            assert isinstance(gen_kw_config, GenKwConfig)

            for keyword in [e.name for e in gen_kw_config.transfer_functions]:
                gen_kw_list.append(f"{key}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword):
                    gen_kw_list.append(f"LOG10_{key}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        gen_kw_keys = self.get_gen_kw()
        all_gen_kw_priors = {}
        for key in gen_kw_keys:
            gen_kw_config = self.config.ensemble_config.parameter_configs[key]
            if isinstance(gen_kw_config, GenKwConfig):
                all_gen_kw_priors[key] = gen_kw_config.get_priors()

        return all_gen_kw_priors

    def get_workflow_job(self, name: str) -> Optional["WorkflowJob"]:
        return self.config.workflow_jobs.get(name)

    def run_ertscript(  # type: ignore
        self,
        ertscript,
        storage: Storage,
        ensemble: Ensemble,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Any:
        return ertscript(self._enkf_main, storage, ensemble=ensemble).run(
            *args, **kwargs
        )

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> "LibresFacade":
        return cls(EnKFMain(ErtConfig.from_file(config_file), read_only))
