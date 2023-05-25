from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

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
from ert.analysis._es_update import ProgressCallback, _get_obs_and_measure_data
from ert.data import MeasuredData
from ert.shared.version import __version__

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iterative_ensemble_smoother import SIES

    from ert._c_wrappers.analysis import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import AnalysisConfig, QueueConfig
    from ert._c_wrappers.enkf.config.gen_kw_config import PriorDict
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs
    from ert._c_wrappers.job_queue import WorkflowJob
    from ert.storage import EnsembleAccessor, EnsembleReader, StorageAccessor


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

    def smoother_update(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._es_update.smootherUpdate(
            prior_storage, posterior_storage, run_id, progress_callback
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

    def set_global_std_scaling(self, weight: float) -> None:
        self._enkf_main.analysisConfig().set_global_std_scaling(weight)

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
            for val in self._enkf_main.ensembleConfig().py_nodes.values()
            if isinstance(val, SurfaceConfig)
        )

    def get_field_parameters(self) -> List[str]:
        return list(
            val.name
            for val in self._enkf_main.ensembleConfig().py_nodes.values()
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
        load_data: bool = True,
        ensemble: Optional[EnsembleReader] = None,
    ) -> MeasuredData:
        return MeasuredData(self, ensemble, keys, index_lists, load_data)

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
        return observation.getImplementationType().name

    def get_data_key_for_obs_key(self, observation_key: str) -> str:
        return self._enkf_main.getObservations()[observation_key].getDataKey()

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

        return ensemble.load_gen_data_as_df(
            [f"{key}@{report_step}"], realizations
        ).droplevel("data_key")

    def load_observation_data(
        self, ensemble: EnsembleReader, keys: Optional[List[str]] = None
    ) -> DataFrame:
        observations = self._enkf_main.getObservations()
        history_length = self._enkf_main.getHistoryLength()
        dates = [
            observations.getObservationTime(index) for index in range(1, history_length)
        ]
        summary_keys = sorted(
            [
                key
                for key in self.get_summary_keys()
                if len(self._enkf_main.ensembleConfig().get_node_observation_keys(key))
                > 0
            ],
            key=lambda k: k.lower(),
        )
        if keys is not None:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist
        columns = summary_keys
        std_columns = [f"STD_{key}" for key in summary_keys]
        df = DataFrame(index=dates, columns=columns + std_columns)
        for key in summary_keys:
            observation_keys = (
                self._enkf_main.ensembleConfig().get_node_observation_keys(key)
            )
            for obs_key in observation_keys:
                observation_data = observations[obs_key]
                for index in range(0, history_length + 1):
                    if observation_data.isActive(index):
                        obs_time = observations.getObservationTime(index)
                        node = observation_data.getNode(index)
                        value = node.getValue()  # type: ignore
                        std = node.getStandardDeviation()  # type: ignore
                        df[key][obs_time] = value
                        df[f"STD_{key}"][obs_time] = std
        return df

    def all_data_type_keys(self) -> List[str]:
        return self.get_summary_keys() + self.gen_kw_keys() + self.get_gen_data_keys()

    def get_all_gen_data_observation_keys(self) -> List[str]:
        return list(
            self._enkf_main.getObservations().getTypedKeylist(
                EnkfObservationImplementationType.GEN_OBS
            )
        )

    def get_all_summary_observation_keys(self) -> List[str]:
        return sorted(
            [
                key
                for key in self.get_summary_keys()
                if len(self._enkf_main.ensembleConfig().get_node_observation_keys(key))
                > 0
            ],
            key=lambda k: k.lower(),
        )

    def observation_keys(self, key: str) -> List[str]:
        if self.is_gen_data_key(key):
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
                    report_step = obs_vector.firstActiveStep()
                    key = obs_vector.getDataKey()

                    if key == data_key and report_step == data_report_step:
                        obs_key = obs_vector.getObservationKey()
            if obs_key is not None:
                return [obs_key]
            else:
                return []
        elif self.is_summary_key(key):
            return [
                str(k)
                for k in self._enkf_main.ensembleConfig().get_node_observation_keys(key)
            ]
        else:
            return []

    def load_all_gen_kw_data(
        self,
        fs: EnsembleReader,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        ens_mask = fs.get_realization_mask_from_state(
            [
                RealizationStateEnum.STATE_INITIALIZED,
                RealizationStateEnum.STATE_HAS_DATA,
            ]
        )
        realizations = [index for index, active in enumerate(ens_mask) if active]

        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization ({realization_index})")
            realizations = [realization_index]

        gen_kw_keys = self.get_gen_kw()
        all_data = {}

        def _flatten(_gen_kw_dict: Dict[str, Any]) -> Dict[str, float]:
            result = {}
            for group, parameters in _gen_kw_dict.items():
                for key, value in parameters.items():
                    combined = f"{group}:{key}"
                    if keys is not None and combined not in keys:
                        continue
                    result[f"{group}:{key}"] = value
            return result

        for realization in realizations:
            realization_data = {}
            for key in gen_kw_keys:
                gen_kw_dict = fs.load_gen_kw_as_dict(key, realization)
                realization_data.update(gen_kw_dict)
            all_data[realization] = _flatten(realization_data)
        gen_kw_df = DataFrame(all_data).T

        gen_kw_df.index.name = "Realization"

        return gen_kw_df

    def gather_gen_kw_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        data = self.load_all_gen_kw_data(
            ensemble,
            [key],
            realization_index=realization_index,
        )
        if key in data:
            return data[key].to_frame().dropna()
        else:
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
        if keys:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist

        try:
            df = ensemble.load_summary_data_as_df(summary_keys, realizations)
        except KeyError:
            return DataFrame()
        df = df.stack().unstack(level=0).swaplevel()
        df.index.names = ["Realization", "Date"]
        return df

    def gather_summary_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        realization_index: Optional[int] = None,
    ) -> Union[DataFrame, Series]:
        data = self.load_all_summary_data(ensemble, [key], realization_index)
        if not data.empty:
            idx = data.index.duplicated()
            if idx.any():
                data = data[~idx]
                _logger.warning(
                    "The simulation data contains duplicate "
                    "timestamps. A possible explanation is that your "
                    "simulation timestep is less than a second."
                )
            data = data.unstack(level="Realization").droplevel("data_key", axis=1)
        return data

    def load_all_misfit_data(self, ensemble: EnsembleReader) -> DataFrame:
        realizations = self.get_active_realizations(ensemble)

        observations = self._enkf_main.getObservations()
        all_observations = [(n.getObsKey(), n.getStepList()) for n in observations]
        measured_data, obs_data = _get_obs_and_measure_data(
            observations, ensemble, all_observations, realizations
        )
        joined = obs_data.join(measured_data, on=["data_key", "axis"], how="inner")
        misfit = DataFrame(index=joined.index)
        for col in measured_data:
            misfit[col] = ((joined["OBS"] - joined[col]) / joined["STD"]) ** 2
        misfit = misfit.groupby("key").sum().T
        misfit.columns = [f"MISFIT:{key}" for key in misfit.columns]
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

    def gather_gen_data_data(
        self,
        ensemble: EnsembleReader,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        key_parts = key.split("@")
        key = key_parts[0]
        if len(key_parts) > 1:
            report_step = int(key_parts[1])
        else:
            report_step = 0

        try:
            data = self.load_gen_data(
                ensemble,
                key,
                report_step,
                realization_index,
            )
        except (ValueError, KeyError):
            data = DataFrame()

        return data.dropna()

    def is_summary_key(self, key: str) -> bool:
        return key in self.get_summary_keys()

    def get_summary_keys(self) -> List[str]:
        return self._enkf_main.ensembleConfig().get_summary_keys()

    def is_gen_kw_key(self, key: str) -> bool:
        return key in self.gen_kw_keys()

    def gen_kw_keys(self) -> List[str]:
        gen_kw_keys = self.get_gen_kw()

        gen_kw_list = []
        for key in gen_kw_keys:
            gen_kw_config = self._enkf_main.ensembleConfig().getNode(key)
            assert isinstance(gen_kw_config, GenKwConfig)

            for keyword in gen_kw_config:
                gen_kw_list.append(f"{key}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword):
                    gen_kw_list.append(f"LOG10_{key}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    def is_gen_data_key(self, key: str) -> bool:
        return key in self.get_gen_data_keys()

    def get_gen_data_keys(self) -> List[str]:
        gen_data_keys = self._enkf_main.ensembleConfig().get_keylist_gen_data()
        gen_data_list = []
        for key in gen_data_keys:
            gen_data_config = self._enkf_main.ensembleConfig().getNodeGenData(key)
            for report_step in gen_data_config.getReportSteps():
                gen_data_list.append(f"{key}@{report_step}")

        return sorted(gen_data_list, key=lambda k: k.lower())

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        gen_kw_keys = self.get_gen_kw()
        all_gen_kw_priors = {}
        for key in gen_kw_keys:
            gen_kw_config = self._enkf_main.ensembleConfig().getNode(key)
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
