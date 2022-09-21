import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
from pandas import DataFrame

from ert._c_wrappers.enkf import EnKFMain, EnkfNode, ErtImplType, ResConfig, RunContext
from ert._c_wrappers.enkf.export import (
    GenDataCollector,
    GenDataObservationCollector,
    GenKwCollector,
    MisfitCollector,
    SummaryCollector,
    SummaryObservationCollector,
)
from ert.analysis import ESUpdate, SmootherSnapshot
from ert.data import MeasuredData

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ecl.grid import EclGrid
    from iterative_ensemble_smoother import IterativeEnsembleSmoother

    from ert._c_wrappers.analysis.analysis_module import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import AnalysisConfig, QueueConfig
    from ert._c_wrappers.enkf.config.gen_kw_config import PriorDict
    from ert._c_wrappers.enkf.enkf_fs import EnkfFs
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs
    from ert._c_wrappers.job_queue import WorkflowJob


class LibresFacade:  # pylint: disable=too-many-public-methods
    """Facade for libres inside ERT."""

    def __init__(self, enkf_main: EnKFMain):
        self._enkf_main = enkf_main
        self._es_update = ESUpdate(enkf_main)

    def write_runpath_list(
        self, iterations: List[int], realizations: List[int]
    ) -> None:
        self._enkf_main.write_runpath_list(iterations, realizations)

    def smoother_update(self, prior_context: RunContext) -> None:
        self._es_update.smootherUpdate(prior_context)

    def iterative_smoother_update(
        self, run_context: RunContext, ies: "IterativeEnsembleSmoother"
    ) -> None:
        self._es_update.iterative_smoother_update(run_context, ies)

    def set_global_std_scaling(self, weight: float) -> None:
        self._enkf_main.analysisConfig().setGlobalStdScaling(weight)

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
        return self._enkf_main.resConfig().model_config.getEnspath()

    @property
    def user_config_file(self) -> Optional[str]:
        return self._enkf_main.resConfig().user_config_file

    @property
    def number_of_iterations(self) -> int:
        return len(self._enkf_main.analysisConfig().getAnalysisIterConfig())

    def get_run_context(self, prior_name: str, target_name: str) -> RunContext:
        fs_manager = self._enkf_main.getEnkfFsManager()
        return RunContext(
            fs_manager.getFileSystem(prior_name),
            fs_manager.getFileSystem(target_name),
        )

    def get_field_parameters(self) -> List[str]:
        return list(
            self._enkf_main.ensembleConfig().getKeylistFromImplType(ErtImplType.FIELD)
        )

    def get_gen_kw(self) -> List[str]:
        return list(
            self._enkf_main.ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
        )

    @property
    def grid_file(self) -> str:
        return self._enkf_main.eclConfig().get_gridfile()

    @property
    def grid(self) -> "EclGrid":
        return self._enkf_main.eclConfig().getGrid()

    def export_field_parameter(
        self, parameter_name: str, case_name: str, filepath: str
    ) -> None:
        file_system = self._enkf_main.getEnkfFsManager().getFileSystem(case_name)
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
        return self._enkf_main.analysisConfig().getModule(module_name)

    def get_ensemble_size(self) -> int:
        return self._enkf_main.getEnsembleSize()

    def get_current_case_name(self) -> str:
        return str(self._enkf_main.getCurrentFileSystem().getCaseName())

    def get_active_realizations(self, case_name: str) -> List[int]:
        fs = self._enkf_main.getFileSystem(case_name, read_only=True)
        return SummaryCollector.createActiveList(fs)

    def case_initialized(self, case: str) -> bool:
        return self._enkf_main.isCaseInitialized(case)

    def get_queue_config(self) -> "QueueConfig":
        return self._enkf_main.get_queue_config()

    def get_number_of_iterations(self) -> int:
        return (
            self._enkf_main.analysisConfig().getAnalysisIterConfig().getNumIterations()
        )

    @property
    def have_observations(self) -> bool:
        return self._enkf_main.have_observations()

    @property
    def run_path(self) -> str:
        return self._enkf_main.getModelConfig().getRunpathAsString()

    def load_from_forward_model(
        self, case: str, realisations: List[bool], iteration: int
    ) -> int:
        fs = self._enkf_main.getFileSystem(case)
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
        return self._enkf_main.getCurrentFileSystem()

    def get_data_key_for_obs_key(self, observation_key: Union[str, int]) -> str:
        return self._enkf_main.getObservations()[observation_key].getDataKey()

    def get_matching_wildcards(self) -> Callable[[str], List[str]]:
        return self._enkf_main.getObservations().getMatchingKeys

    def get_observation_key(self, index: Union[str, int]) -> str:
        return self._enkf_main.getObservations()[index].getKey()

    def load_gen_data(self, case_name: str, key: str, report_step: int) -> DataFrame:
        return GenDataCollector.loadGenData(
            self._enkf_main, case_name, key, report_step
        )

    def load_observation_data(
        self, case_name: str, keys: Optional[List[str]] = None
    ) -> DataFrame:
        return SummaryObservationCollector.loadObservationData(
            self._enkf_main, case_name, keys
        )

    def select_or_create_new_case(self, case_name: str) -> None:
        if self.get_current_case_name() != case_name:
            fs = self._enkf_main.getFileSystem(case_name)
            self._enkf_main.switchFileSystem(fs)

    def cases(self) -> List[str]:
        return self._enkf_main.getCaseList()

    def case_has_data(self, case: str) -> bool:
        return self._enkf_main.caseHasData(case)

    def all_data_type_keys(self) -> List[str]:
        return self._enkf_main.getKeyManager().allDataTypeKeys()

    def get_all_gen_data_observation_keys(self) -> List[str]:
        return GenDataObservationCollector.getAllObservationKeys(self._enkf_main)

    def get_all_summary_observation_keys(self) -> List[str]:
        return SummaryObservationCollector.getAllObservationKeys(self._enkf_main)

    def observation_keys(self, key: str) -> List[str]:
        if self._enkf_main.getKeyManager().isGenDataKey(key):
            key_parts = key.split("@")
            key = key_parts[0]
            if len(key_parts) > 1:
                report_step = int(key_parts[1])
            else:
                report_step = 0

            obs_key = GenDataObservationCollector.getObservationKeyForDataKey(
                self._enkf_main, key, report_step
            )
            if obs_key is not None:
                return [obs_key]
            else:
                return []
        elif self._enkf_main.getKeyManager().isSummaryKey(key):
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
        return GenKwCollector.loadAllGenKwData(
            self._enkf_main,
            case_name=case_name,
            keys=keys,
            realization_index=realization_index,
        )

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
        return SummaryCollector.loadAllSummaryData(
            self._enkf_main,
            case_name=case_name,
            keys=keys,
            realization_index=realization_index,
        )

    def gather_summary_data(
        self,
        case: str,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
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
        return MisfitCollector.loadAllMisfitData(self._enkf_main, case_name=case_name)

    def refcase_data(self, key: str) -> DataFrame:
        refcase = self._enkf_main.eclConfig().getRefcase()

        if refcase is None or key not in refcase:
            return DataFrame()

        values = refcase.numpy_vector(key, report_only=False)
        dates = refcase.numpy_dates

        data = DataFrame(zip(dates, values), columns=["Date", key])
        data.set_index("Date", inplace=True)

        return data.iloc[1:]

    def history_data(self, key: str, case: Optional[str] = None) -> DataFrame:
        if not self.is_summary_key(key):
            return DataFrame()

        if case is None:
            return self.refcase_data(key)

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
            data = GenDataCollector.loadGenData(
                self._enkf_main,
                case,
                key,
                report_step,
                realization_index,
            )
        except (ValueError, KeyError):
            data = DataFrame()

        return data.dropna()  # removes all rows that has a NaN

    def is_summary_key(self, key: str) -> bool:
        return key in self._enkf_main.getKeyManager().summaryKeys()

    def get_summary_keys(self) -> List[str]:
        return self._enkf_main.getKeyManager().summaryKeys()

    def is_gen_kw_key(self, key: str) -> bool:
        return key in self._enkf_main.getKeyManager().genKwKeys()

    def gen_kw_keys(self) -> List[str]:
        return self._enkf_main.getKeyManager().genKwKeys()

    def is_gen_data_key(self, key: str) -> bool:
        return key in self._enkf_main.getKeyManager().genDataKeys()

    def get_gen_data_keys(self) -> List[str]:
        return self._enkf_main.getKeyManager().genDataKeys()

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        return self._enkf_main.getKeyManager().gen_kw_priors()

    @property
    def update_snapshots(self) -> Dict[str, SmootherSnapshot]:
        return self._es_update.update_snapshots

    def get_alpha(self) -> float:
        return self._enkf_main.analysisConfig().getEnkfAlpha()

    def get_std_cutoff(self) -> float:
        return self._enkf_main.analysisConfig().getStdCutoff()

    def get_workflow_job(self, name: str) -> Optional["WorkflowJob"]:
        if self._enkf_main.getWorkflowList().hasJob(name):
            return self._enkf_main.getWorkflowList().getJob(name)
        return None

    def run_ertscript(self, ertscript, *args, **kwargs):  # type: ignore
        return ertscript(self._enkf_main).run(*args, **kwargs)

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> "LibresFacade":
        return cls(EnKFMain(ResConfig(config_file), read_only))
