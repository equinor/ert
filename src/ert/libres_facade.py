import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from pandas import DataFrame

from ert.analysis import ESUpdate, SmootherSnapshot
from res.enkf import EnKFMain
from res.enkf.export import (
    GenDataCollector,
    GenDataObservationCollector,
    GenKwCollector,
    SummaryCollector,
    SummaryObservationCollector,
)
from res.enkf.plot_data import PlotBlockDataLoader

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:

    from ert.analysis import ModuleData
    from res.analysis.analysis_module import AnalysisModule
    from res.enkf import AnalysisConfig, ObsVector, QueueConfig, RunContext
    from res.enkf.config.gen_kw_config import PriorDict
    from res.enkf.enkf_fs import EnkfFs
    from res.enkf.enkf_obs import EnkfObs
    from res.job_queue import WorkflowJob


class LibresFacade:
    """Facade for libres inside ERT."""

    def __init__(self, enkf_main: EnKFMain):
        self._enkf_main = enkf_main
        self._es_update = ESUpdate(enkf_main)

    def smoother_update(self, prior_context: "RunContext") -> None:
        self._es_update.smootherUpdate(prior_context)

    def iterative_smoother_update(
        self, run_context: "RunContext", module_data: "ModuleData"
    ) -> None:
        self._es_update.iterative_smoother_update(run_context, module_data)

    def set_global_std_scaling(self, weight: float) -> None:
        self._enkf_main.analysisConfig().setGlobalStdScaling(weight)

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
        realizations = SummaryCollector.createActiveList(fs)

        return realizations

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

    def load_observation_data(
        self, case_name: str, keys: Optional[List[str]] = None
    ) -> DataFrame:
        return SummaryObservationCollector.loadObservationData(
            self._enkf_main, case_name, keys
        )

    def create_plot_block_data_loader(
        self, obs_vector: "ObsVector"
    ) -> PlotBlockDataLoader:
        return PlotBlockDataLoader(obs_vector)

    def select_or_create_new_case(self, case_name: str) -> None:
        if self.get_current_case_name() != case_name:
            fs = self._enkf_main.getFileSystem(case_name)
            self._enkf_main.switchFileSystem(fs)

    def cases(self) -> List[str]:
        return self._enkf_main.getCaseList()

    def is_case_hidden(self, case: str) -> bool:
        return self._enkf_main.isCaseHidden(case)

    def case_has_data(self, case: str) -> bool:
        return self._enkf_main.caseHasData(case)

    def is_case_running(self, case: str) -> bool:
        return self._enkf_main.isCaseRunning(case)

    def all_data_type_keys(self) -> List[str]:
        return self._enkf_main.getKeyManager().allDataTypeKeys()

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

    def gather_gen_kw_data(
        self,
        case: str,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        data = GenKwCollector.loadAllGenKwData(
            self._enkf_main,
            case,
            [key],
            realization_index=realization_index,
        )
        if key in data:
            return data[key].to_frame().dropna()
        else:
            return DataFrame()

    def gather_summary_data(
        self,
        case: str,
        key: str,
        realization_index: Optional[int] = None,
    ) -> DataFrame:
        data = SummaryCollector.loadAllSummaryData(
            self._enkf_main, case, [key], realization_index
        )
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
        self,
        case: str,
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
