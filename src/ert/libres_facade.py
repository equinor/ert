from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from deprecation import deprecated
from ecl.grid import EclGrid
from pandas import DataFrame, Series

from ert._c_wrappers.enkf import EnKFMain, EnsembleConfig, ErtConfig
from ert._c_wrappers.enkf.config import GenKwConfig
from ert._c_wrappers.enkf.config.field_config import Field
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig
from ert._c_wrappers.enkf.enums import (
    EnkfObservationImplementationType,
    RealizationStateEnum,
)
from ert.analysis import ESUpdate, SmootherSnapshot
from ert.analysis._es_update import ProgressCallback
from ert.data import MeasuredData
from ert.data._measured_data import ResponseError
from ert.shared.version import __version__
from ert.storage import EnsembleReader

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iterative_ensemble_smoother import SIES

    from ert._c_wrappers.analysis import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import AnalysisConfig, QueueConfig
    from ert._c_wrappers.enkf.config.gen_kw_config import PriorDict
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs
    from ert.job_queue import WorkflowJob
    from ert.storage import EnsembleAccessor, StorageAccessor


class LibresFacade:  # pylint: disable=too-many-public-methods
    """The intention of this class is to expose properties or data of ert
    commonly used in other project. It is part of the public interface of ert,
    and as such changes here should not be taken lightly."""

    def __init__(self, enkf_main: EnKFMain):
        self._enkf_main = enkf_main
        self._es_update = ESUpdate(enkf_main)

    def write_runpath_list(
        self, iterations: List[int], realizations: List[int]
    ) -> None:
        self._enkf_main.write_runpath_list(iterations, realizations)

    def smoother_update(  # pylint: disable=too-many-arguments
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
        global_std_scaling: float = 1.0,
    ) -> None:
        self._es_update.smootherUpdate(
            prior_storage,
            posterior_storage,
            run_id,
            progress_callback,
            global_std_scaling,
        )

    # pylint: disable-msg=too-many-arguments
    def iterative_smoother_update(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        ies: "SIES",
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._es_update.iterative_smoother_update(
            prior_storage, posterior_storage, ies, run_id, progress_callback
        )

    def set_log_path(self, output_path: str) -> None:
        self._enkf_main.analysisConfig().set_log_path(output_path)

    @property
    def update_configuration(self) -> "UpdateConfiguration":
        return self._enkf_main.update_configuration

    @update_configuration.setter
    def update_configuration(self, value: Any) -> None:
        self._enkf_main.update_configuration = value

    @property
    def enspath(self) -> str:
        return self._enkf_main.resConfig().ens_path

    @property
    def user_config_file(self) -> Optional[str]:
        return self._enkf_main.resConfig().user_config_file

    @property
    def number_of_iterations(self) -> int:
        return self._enkf_main.analysisConfig().num_iterations

    def get_surface_parameters(self) -> List[str]:
        return list(
            val.name
            for val in self._enkf_main.ensembleConfig().parameter_configuration
            if isinstance(val, SurfaceConfig)
        )

    def get_field_parameters(self) -> List[str]:
        return list(
            val.name
            for val in self._enkf_main.ensembleConfig().parameter_configuration
            if isinstance(val, Field)
        )

    def get_gen_kw(self) -> List[str]:
        return self._enkf_main.ensembleConfig().get_keylist_gen_kw()

    @property
    def grid_file(self) -> Optional[str]:
        return self._enkf_main.ensembleConfig().grid_file

    @property
    @deprecated(
        deprecated_in="5.0",
        current_version=__version__,
        details=(
            "Grid's statefullness is linked to the configuration and not stored data.\n"
            "This can lead to inconsistencies between field and grid."
        ),
    )  # type: ignore
    def grid(self) -> Optional[EclGrid]:
        path = self.grid_file
        if path:
            ext = Path(path).suffix
            if ext in [".EGRID", ".GRID"]:
                return EclGrid.load_from_file(path)
        return None

    @property
    def ensemble_config(self) -> EnsembleConfig:
        return self._enkf_main.ensembleConfig()

    def get_measured_data(  # pylint: disable=too-many-arguments
        self,
        keys: List[str],
        index_lists: Optional[List[List[int]]] = None,
        ensemble: Optional[EnsembleReader] = None,
    ) -> MeasuredData:
        assert isinstance(ensemble, EnsembleReader)
        return MeasuredData(self, ensemble, keys, index_lists)

    def get_analysis_config(self) -> "AnalysisConfig":
        return self._enkf_main.analysisConfig()

    def get_analysis_module(self, module_name: str) -> "AnalysisModule":
        return self._enkf_main.analysisConfig().get_module(module_name)

    def get_ensemble_size(self) -> int:
        return self._enkf_main.getEnsembleSize()

    def get_active_realizations(self, ensemble: EnsembleReader) -> List[int]:
        return ensemble.realization_list(RealizationStateEnum.STATE_HAS_DATA)

    def get_queue_config(self) -> "QueueConfig":
        return self._enkf_main.get_queue_config()

    def get_number_of_iterations(self) -> int:
        return self._enkf_main.analysisConfig().num_iterations

    @property
    def have_smoother_parameters(self) -> bool:
        return bool(self._enkf_main.ensembleConfig().parameters)

    @property
    def have_observations(self) -> bool:
        return self._enkf_main.have_observations()

    @property
    def run_path(self) -> str:
        return self._enkf_main.getModelConfig().runpath_format_string

    def get_run_paths(self, realizations: List[int], iteration: int) -> List[str]:
        run_paths = self._enkf_main.runpaths.get_paths(realizations, iteration)
        return list(run_paths)

    def load_from_forward_model(
        self, ensemble: EnsembleAccessor, realisations: List[bool], iteration: int
    ) -> int:
        return self._enkf_main.loadFromForwardModel(realisations, iteration, ensemble)

    def get_observations(self) -> "EnkfObs":
        return self._enkf_main.getObservations()

    def get_impl_type_name_for_obs_key(self, key: str) -> str:
        observation = self._enkf_main.getObservations()[key]
        return observation.observation_type.name

    def get_data_key_for_obs_key(self, observation_key: str) -> str:
        obs = self._enkf_main.getObservations()[observation_key]
        if obs.observation_type == EnkfObservationImplementationType.SUMMARY_OBS:
            return list(obs.observations.values())[0].summary_key  # type: ignore
        else:
            return obs.data_key

    def get_matching_wildcards(self) -> Callable[[str], List[str]]:
        return self._enkf_main.getObservations().getMatchingKeys

    def load_gen_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        report_step: int,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        realizations = ensemble.realization_list(RealizationStateEnum.STATE_HAS_DATA)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]
        try:
            vals = ensemble.load_response(key, tuple(realizations)).sel(
                report_step=report_step, drop=True
            )
        except KeyError:
            raise KeyError(f"Missing response: {key}")
        index = pd.Index(vals.index.values, name="axis")
        return pd.DataFrame(
            data=vals["values"].values.reshape(len(vals.realization), -1).T,
            index=index,
            columns=realizations,
        )

    def all_data_type_keys(self) -> List[str]:
        return self.get_summary_keys() + self.gen_kw_keys() + self.get_gen_data_keys()

    def observation_keys(self, key: str) -> List[str]:
        if key in self.get_gen_data_keys():
            key_parts = key.split("@")
            data_key = key_parts[0]
            if len(key_parts) > 1:
                data_report_step = int(key_parts[1])
            else:
                data_report_step = 0

            obs_key = None

            enkf_obs = self._enkf_main.getObservations()
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

    def load_all_gen_kw_data(
        self,
        ensemble: EnsembleReader,
        group: Optional[str] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads all GEN_KW data into a DataFrame.

        This function retrieves GEN_KW data from the given ensemble reader.
        index and returns it in a pandas DataFrame.

        Args:
            ensemble: The ensemble reader from which to load the GEN_KW data.

        Returns:
            DataFrame: A pandas DataFrame containing the GEN_KW data.

        Raises:
            IndexError: If a non-existent realization index is provided.

        Note:
            Any provided keys that are not gen_kw will be ignored.
        """
        ens_mask = ensemble.get_realization_mask_from_state(
            [
                RealizationStateEnum.STATE_INITIALIZED,
                RealizationStateEnum.STATE_HAS_DATA,
            ]
        )
        realizations = (
            [realization_index]
            if realization_index is not None
            else [index for index, active in enumerate(ens_mask) if active]
        )

        dataframes = []
        gen_kws = [
            config
            for config in ensemble.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            try:
                ds = ensemble.load_parameters(
                    key.name, realizations, var="transformed_values"
                )
                ds["names"] = np.char.add(f"{key.name}:", ds["names"].astype(np.str_))
                df = ds.to_dataframe().unstack(level="names")
                df.columns = df.columns.droplevel()
                for parameter in df.columns:
                    if key.shouldUseLogScale(parameter.split(":")[1]):
                        df[f"LOG10_{parameter}"] = np.log10(df[parameter])
                dataframes.append(df)
            except KeyError:
                pass
        if not dataframes:
            return pd.DataFrame()

        # Format the DataFrame in a way that old code expects it
        dataframe = pd.concat(dataframes, axis=1)
        dataframe.columns.name = None
        dataframe.index.name = "Realization"

        return dataframe.sort_index(axis=1)

    def gather_gen_kw_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        realization_index: Optional[int],
    ) -> DataFrame:
        try:
            data = self.load_all_gen_kw_data(
                ensemble,
                key.split(":")[0],
                realization_index,
            )
            return data[key].to_frame().dropna()
        except KeyError:
            return DataFrame()

    def load_all_summary_data(
        self,
        ensemble: EnsembleReader,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        realizations = self.get_active_realizations(ensemble)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = ensemble.get_summary_keyset()

        try:
            df = ensemble.load_response("summary", tuple(realizations)).to_dataframe()
        except (ValueError, KeyError):
            return pd.DataFrame()
        df = df.unstack(level="name")
        df.columns = [col[1] for col in df.columns.values]
        df.index = df.index.rename(
            {"time": "Date", "realization": "Realization"}
        ).reorder_levels(["Realization", "Date"])
        if keys:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist
            return df[summary_keys]
        return df

    def gather_summary_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        realization_index: Optional[int] = None,
    ) -> Union[DataFrame, Series]:
        data = self.load_all_summary_data(ensemble, [key], realization_index)
        if data.empty:
            return data
        idx = data.index.duplicated()
        if idx.any():
            data = data[~idx]
            _logger.warning(
                "The simulation data contains duplicate "
                "timestamps. A possible explanation is that your "
                "simulation timestep is less than a second."
            )
        return data.unstack(level="Realization")

    def load_all_misfit_data(self, ensemble: EnsembleReader) -> DataFrame:
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
            measured_data = self.get_measured_data(
                self._enkf_main._observation_keys, ensemble=ensemble
            )
        except ResponseError:
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

        return misfit

    def refcase_data(self, key: str) -> DataFrame:
        refcase = self._enkf_main.ensembleConfig().refcase

        if refcase is None or key not in refcase:
            return DataFrame()

        values = refcase.numpy_vector(key, report_only=False)
        dates = refcase.numpy_dates

        data = DataFrame(zip(dates, values), columns=["Date", key])
        data.set_index("Date", inplace=True)

        return data.iloc[1:]

    def history_data(
        self, key: str, ensemble: Optional[EnsembleReader] = None
    ) -> Union[DataFrame, Series]:
        if ensemble is None:
            return self.refcase_data(key)

        if key not in ensemble.get_summary_keyset():
            return DataFrame()

        data = self.gather_summary_data(ensemble, key)
        if data.empty and ensemble is not None:
            data = self.refcase_data(key)

        return data

    def get_summary_keys(self) -> List[str]:
        return self._enkf_main.ensembleConfig().get_summary_keys()

    def gen_kw_keys(self) -> List[str]:
        gen_kw_keys = self.get_gen_kw()

        gen_kw_list = []
        for key in gen_kw_keys:
            gen_kw_config = self._enkf_main.ensembleConfig().parameter_configs[key]
            assert isinstance(gen_kw_config, GenKwConfig)

            for keyword in [e.name for e in gen_kw_config.transfer_functions]:
                gen_kw_list.append(f"{key}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword):
                    gen_kw_list.append(f"LOG10_{key}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    def get_gen_data_keys(self) -> List[str]:
        gen_data_keys = self._enkf_main.ensembleConfig().get_keylist_gen_data()
        gen_data_list = []
        for key in gen_data_keys:
            gen_data_config = self._enkf_main.ensembleConfig().getNodeGenData(key)
            for report_step in gen_data_config.report_steps:
                gen_data_list.append(f"{key}@{report_step}")

        return sorted(gen_data_list, key=lambda k: k.lower())

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        gen_kw_keys = self.get_gen_kw()
        all_gen_kw_priors = {}
        for key in gen_kw_keys:
            gen_kw_config = self._enkf_main.ensembleConfig().parameter_configs[key]
            if isinstance(gen_kw_config, GenKwConfig):
                all_gen_kw_priors[key] = gen_kw_config.get_priors()

        return all_gen_kw_priors

    @property
    def update_snapshots(self) -> Dict[str, SmootherSnapshot]:
        return self._es_update.update_snapshots

    def get_alpha(self) -> float:
        return self._enkf_main.analysisConfig().get_enkf_alpha()

    def get_std_cutoff(self) -> float:
        return self._enkf_main.analysisConfig().get_std_cutoff()

    def get_workflow_job(self, name: str) -> Optional["WorkflowJob"]:
        return self._enkf_main.resConfig().workflow_jobs.get(name)

    def run_ertscript(  # type: ignore
        self,
        ertscript,
        storage: StorageAccessor,
        ensemble: EnsembleAccessor,
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
