import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
from ecl.grid import EclGrid
from pandas import DataFrame, MultiIndex, Series

from ert import _clib
from ert._c_wrappers.enkf import EnKFMain, EnkfNode, ErtConfig, ErtImplType
from ert._c_wrappers.enkf.config import GenKwConfig
from ert._c_wrappers.enkf.enums import (
    EnkfObservationImplementationType,
    RealizationStateEnum,
)
from ert.analysis import ESUpdate, SmootherSnapshot
from ert.data import MeasuredData

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iterative_ensemble_smoother import IterativeEnsembleSmoother

    from ert._c_wrappers.analysis import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import AnalysisConfig, QueueConfig
    from ert._c_wrappers.enkf.config.gen_kw_config import PriorDict
    from ert._c_wrappers.enkf.enkf_fs import EnkfFs
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs
    from ert._c_wrappers.job_queue import WorkflowJob


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
        self, prior_storage: "EnkfFs", posterior_storage: "EnkfFs", run_id: str
    ) -> None:
        self._es_update.smootherUpdate(prior_storage, posterior_storage, run_id)

    def iterative_smoother_update(
        self,
        prior_storage: "EnkfFs",
        posterior_storage: "EnkfFs",
        ies: "IterativeEnsembleSmoother",
        run_id: str,
    ) -> None:
        self._es_update.iterative_smoother_update(
            prior_storage, posterior_storage, ies, run_id
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

    def get_field_parameters(self) -> List[str]:
        return list(
            self._enkf_main.ensembleConfig().getKeylistFromImplType(ErtImplType.FIELD)
        )

    def get_gen_kw(self) -> List[str]:
        return list(
            self._enkf_main.ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
        )

    @property
    def grid_file(self) -> Optional[str]:
        return self._enkf_main.ensembleConfig().grid_file

    @property
    def grid(self) -> Optional[EclGrid]:
        return self._enkf_main.ensembleConfig().grid

    def export_field_parameter(
        self, parameter_name: str, case_name: str, filepath: str
    ) -> None:
        file_system = self._enkf_main.storage_manager[case_name]
        config_node = self._enkf_main.ensembleConfig()[parameter_name]
        ext = config_node.get_enkf_outfile().rsplit(".")[-1]
        EnkfNode.exportMany(
            config_node,
            filepath + "." + ext,
            file_system,
            np.arange(0, self.get_ensemble_size()),
        )

    def get_measured_data(  # pylint: disable=too-many-arguments
        self,
        keys: List[str],
        index_lists: Optional[List[List[int]]] = None,
        load_data: bool = True,
        case_name: Optional[str] = None,
    ) -> MeasuredData:
        return MeasuredData(self, keys, index_lists, load_data, case_name)

    def get_analysis_config(self) -> "AnalysisConfig":
        return self._enkf_main.analysisConfig()

    def get_analysis_module(self, module_name: str) -> "AnalysisModule":
        return self._enkf_main.analysisConfig().get_module(module_name)

    def get_ensemble_size(self) -> int:
        return self._enkf_main.getEnsembleSize()

    def get_current_case_name(self) -> str:
        return str(self.get_current_fs().case_name)

    def get_active_realizations(self, case_name: str) -> List[int]:
        fs = self._enkf_main.storage_manager[case_name]
        state_map = fs.getStateMap()
        ens_mask = state_map.selectMatching(RealizationStateEnum.STATE_HAS_DATA)
        return [index for index, element in enumerate(ens_mask) if element]

    def case_initialized(self, case: str) -> bool:
        if case in self._enkf_main.storage_manager:
            return self._enkf_main.storage_manager[case].is_initalized
        else:
            return False

    def get_queue_config(self) -> "QueueConfig":
        return self._enkf_main.get_queue_config()

    def get_number_of_iterations(self) -> int:
        return self._enkf_main.analysisConfig().num_iterations

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
        self, case: str, realisations: List[bool], iteration: int
    ) -> int:
        fs = self._enkf_main.storage_manager[case]
        return self._enkf_main.loadFromForwardModel(realisations, iteration, fs)

    def get_observations(self) -> "EnkfObs":
        return self._enkf_main.getObservations()

    def get_impl_type_name_for_obs_key(self, key: str) -> str:
        observation = self._enkf_main.getObservations()[key]
        return observation.getImplementationType().name  # type: ignore

    def get_impl_type_name_for_ensemble_config_node(self, name: str) -> str:
        node = self._enkf_main.ensembleConfig().getNode(name)
        return node.getImplementationType().name  # type: ignore

    def get_data_size_for_ensemble_config_node(self, name: str) -> int:
        return (
            self._enkf_main.ensembleConfig()
            .getNode(name)
            .getFieldModelConfig()
            .get_data_size()
        )

    def get_current_fs(self) -> "EnkfFs":
        return self._enkf_main.storage_manager.current_case

    def get_data_key_for_obs_key(self, observation_key: Union[str, int]) -> str:
        return self._enkf_main.getObservations()[observation_key].getDataKey()

    def get_matching_wildcards(self) -> Callable[[str], List[str]]:
        return self._enkf_main.getObservations().getMatchingKeys

    def get_observation_key(self, index: Union[str, int]) -> str:
        return self._enkf_main.getObservations()[index].getKey()

    def load_gen_data(
        self,
        case_name: str,
        key: str,
        report_step: int,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        fs = self._enkf_main.storage_manager[case_name]
        realizations = fs.realizationList(RealizationStateEnum.STATE_HAS_DATA)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]
        config_node = self._enkf_main.ensembleConfig().getNode(key)
        if report_step not in config_node.getDataModelConfig().getReportSteps():
            raise ValueError(
                f"No report step {report_step} in report steps: "
                f"{config_node.getDataModelConfig().getReportSteps()}"
            )
        # pylint: disable=c-extension-no-member
        data_array = _clib.enkf_fs_general_data.gendata_get_realizations(
            config_node, fs, realizations, report_step
        )
        return DataFrame(data=data_array, columns=np.array(realizations))

    def load_observation_data(
        self, case_name: str, keys: Optional[List[str]] = None
    ) -> DataFrame:
        observations = self._enkf_main.getObservations()
        history_length = self._enkf_main.getHistoryLength()
        dates = [
            observations.getObservationTime(index).datetime()
            for index in range(1, history_length)
        ]
        summary_keys = sorted(
            [
                key
                for key in self.get_summary_keys()
                if len(
                    self._enkf_main.ensembleConfig().getNode(key).getObservationKeys()
                )
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
                self._enkf_main.ensembleConfig().getNode(key).getObservationKeys()
            )
            for obs_key in observation_keys:
                observation_data = observations[obs_key]
                for index in range(0, history_length + 1):
                    if observation_data.isActive(index):
                        obs_time = observations.getObservationTime(index).datetime()
                        node = observation_data.getNode(index)
                        value = node.getValue()  # type: ignore
                        std = node.getStandardDeviation()  # type: ignore
                        df[key][obs_time] = value
                        df[f"STD_{key}"][obs_time] = std
        return df

    def select_or_create_new_case(self, case_name: str) -> "EnkfFs":
        if case_name not in self._enkf_main.storage_manager:
            fs = self._enkf_main.storage_manager.add_case(case_name)
        else:
            fs = self._enkf_main.storage_manager[case_name]
        if self.get_current_case_name() != case_name:
            self._enkf_main.switchFileSystem(fs.case_name)
        return fs

    def cases(self) -> List[str]:
        def sort_key(s: str) -> List[Union[int, str]]:
            _nsre = re.compile("([0-9]+)")
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s)
            ]

        return sorted(self._enkf_main.storage_manager.cases, key=sort_key)

    def all_data_type_keys(self) -> List[str]:
        return self.get_summary_keys() + self.gen_kw_keys() + self.get_gen_data_keys()

    def get_all_gen_data_observation_keys(self) -> List[str]:
        return list(
            self._enkf_main.getObservations().getTypedKeylist(
                EnkfObservationImplementationType.GEN_OBS  # type: ignore
            )
        )

    def get_all_summary_observation_keys(self) -> List[str]:
        return sorted(
            [
                key
                for key in self.get_summary_keys()
                if len(
                    self._enkf_main.ensembleConfig().getNode(key).getObservationKeys()
                )
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
                for k in self._enkf_main.ensembleConfig()
                .getNode(key)
                .getObservationKeys()
            ]
        else:
            return []

    def load_all_gen_kw_data(
        self,
        case_name: str,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        fs = self._enkf_main.storage_manager[case_name]

        ens_mask = fs.getStateMap().selectMatching(
            RealizationStateEnum.STATE_INITIALIZED
            | RealizationStateEnum.STATE_HAS_DATA,
        )
        realizations = [index for index, active in enumerate(ens_mask) if active]

        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization ({realization_index})")
            realizations = [realization_index]

        gen_kw_keys = self.gen_kw_keys()

        if keys is not None:
            gen_kw_keys = [
                key for key in keys if key in gen_kw_keys
            ]  # ignore keys that doesn't exist

        # pylint: disable=c-extension-no-member
        gen_kw_array = _clib.enkf_fs_keyword_data.keyword_data_get_realizations(
            self._enkf_main.ensembleConfig(), fs, gen_kw_keys, realizations
        )
        gen_kw_data = DataFrame(
            data=gen_kw_array, index=realizations, columns=gen_kw_keys
        )

        gen_kw_data.index.name = "Realization"
        return gen_kw_data

    def gather_gen_kw_data(
        self,
        case: str,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        data = self.load_all_gen_kw_data(
            case,
            [key],
            realization_index=realization_index,
        )
        if key in data:
            return data[key].to_frame().dropna()
        else:
            return DataFrame()

    def load_all_summary_data(
        self,
        case_name: str,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        fs = self._enkf_main.storage_manager[case_name]

        time_map = fs.getTimeMap()
        dates = [time_map[index] for index in range(1, len(time_map))]

        realizations = self.get_active_realizations(case_name)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = fs.getSummaryKeySet()
        if keys:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist
        # pylint: disable=c-extension-no-member
        summary_data = _clib.enkf_fs_summary_data.get_summary_data(
            fs, summary_keys, realizations, len(dates)
        )

        if np.isnan(summary_data).all():
            return DataFrame()

        multi_index = MultiIndex.from_product(
            [realizations, dates], names=["Realization", "Date"]
        )

        return DataFrame(data=summary_data, index=multi_index, columns=summary_keys)

    def gather_summary_data(
        self,
        case: str,
        key: str,
        realization_index: Optional[int] = None,
    ) -> Union[DataFrame, Series]:
        data = self.load_all_summary_data(case, [key], realization_index)
        if not data.empty:
            idx = data.index.duplicated()
            if idx.any():
                data = data[~idx]
                _logger.warning(
                    "The simulation data contains duplicate "
                    "timestamps. A possible explanation is that your "
                    "simulation timestep is less than a second."
                )
            data = data.unstack(level="Realization").droplevel(0, axis=1)
        return data

    def load_all_misfit_data(self, case_name: str) -> DataFrame:
        realizations = self.get_active_realizations(case_name)
        fs = self._enkf_main.storage_manager[case_name]
        misfit_keys = []
        for obs_vector in self._enkf_main.getObservations():
            misfit_keys.append(f"MISFIT:{obs_vector.getObservationKey()}")

        misfit_keys.append("MISFIT:TOTAL")

        misfit_sum_index = len(misfit_keys) - 1

        misfit_array = np.empty(
            shape=(len(misfit_keys), len(realizations)), dtype=np.float64
        )
        misfit_array.fill(np.nan)
        misfit_array[misfit_sum_index] = 0.0

        for column_index, obs_vector in enumerate(self._enkf_main.getObservations()):
            for realization_index, realization_number in enumerate(realizations):
                misfit = obs_vector.getTotalChi2(
                    fs,
                    realization_number,
                )

                misfit_array[column_index][realization_index] = misfit
                misfit_array[misfit_sum_index][realization_index] += misfit

        misfit_data = DataFrame(
            data=np.transpose(misfit_array), index=realizations, columns=misfit_keys
        )
        misfit_data.index.name = "Realization"

        return misfit_data

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
        self, key: str, case: Optional[str] = None
    ) -> Union[DataFrame, Series]:
        if case is None:
            return self.refcase_data(key)

        storage = self._enkf_main.storage_manager[case]
        if key not in storage.getSummaryKeySet():
            return DataFrame()

        data = self.gather_summary_data(case, key)
        if data.empty and case is not None:
            data = self.refcase_data(key)

        return data

    def gather_gen_data_data(
        self, case: str, key: str, realization_index: Optional[int] = None
    ) -> DataFrame:
        key_parts = key.split("@")
        key = key_parts[0]
        if len(key_parts) > 1:
            report_step = int(key_parts[1])
        else:
            report_step = 0

        try:
            data = self.load_gen_data(
                case,
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
        return sorted(
            list(
                self._enkf_main.ensembleConfig().getKeylistFromImplType(
                    ErtImplType.SUMMARY
                )
            ),
            key=lambda k: k.lower(),
        )

    def is_gen_kw_key(self, key: str) -> bool:
        return key in self.gen_kw_keys()

    def gen_kw_keys(self) -> List[str]:
        gen_kw_keys = self.get_gen_kw()

        gen_kw_list = []
        for key in gen_kw_keys:
            enkf_config_node = self._enkf_main.ensembleConfig().getNode(key)
            gen_kw_config = enkf_config_node.getModelConfig()
            assert isinstance(gen_kw_config, GenKwConfig)

            for keyword_index, keyword in enumerate(gen_kw_config):
                gen_kw_list.append(f"{key}:{keyword}")

                if gen_kw_config.shouldUseLogScale(keyword_index):
                    gen_kw_list.append(f"LOG10_{key}:{keyword}")

        return sorted(gen_kw_list, key=lambda k: k.lower())

    def is_gen_data_key(self, key: str) -> bool:
        return key in self.get_gen_data_keys()

    def get_gen_data_keys(self) -> List[str]:
        gen_data_keys = self._enkf_main.ensembleConfig().getKeylistFromImplType(
            ErtImplType.GEN_DATA
        )
        gen_data_list = []
        for key in gen_data_keys:
            enkf_config_node = self._enkf_main.ensembleConfig().getNode(key)
            gen_data_config = enkf_config_node.getDataModelConfig()

            for report_step in gen_data_config.getReportSteps():
                gen_data_list.append(f"{key}@{report_step}")

        return sorted(gen_data_list, key=lambda k: k.lower())

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        gen_kw_keys = self.get_gen_kw()
        all_gen_kw_priors = {}
        for key in gen_kw_keys:
            enkf_config_node = self._enkf_main.ensembleConfig().getNode(key)
            gen_kw_config = enkf_config_node.getModelConfig()
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

    def run_ertscript(self, ertscript, *args, **kwargs):  # type: ignore
        return ertscript(self._enkf_main).run(*args, **kwargs)

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> "LibresFacade":
        return cls(EnKFMain(ErtConfig.from_file(config_file), read_only))
