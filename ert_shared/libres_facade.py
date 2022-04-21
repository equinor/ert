from typing import List

import logging

from pandas import DataFrame

from res.analysis.analysis_module import AnalysisModule
from res.enkf.export import (
    GenDataCollector,
    SummaryCollector,
    SummaryObservationCollector,
    GenDataObservationCollector,
    GenKwCollector,
)
from res.enkf.plot_data import PlotBlockDataLoader
from res.enkf import EnKFMain

_logger = logging.getLogger(__name__)


class LibresFacade:
    """Facade for libres inside ERT."""

    def __init__(self, enkf_main: EnKFMain):
        self._enkf_main = enkf_main

    def get_analysis_config(self):
        return self._enkf_main.analysisConfig()

    def get_analysis_module(self, module_name: str) -> AnalysisModule:
        return self._enkf_main.analysisConfig().getModule(module_name)

    def get_ensemble_size(self):
        return self._enkf_main.getEnsembleSize()

    def get_current_case_name(self):
        return str(
            self._enkf_main.getEnkfFsManager().getCurrentFileSystem().getCaseName()
        )

    def get_active_realizations(self, case_name):
        fs = self._enkf_main.getEnkfFsManager().getFileSystem(case_name)
        realizations = SummaryCollector.createActiveList(self._enkf_main, fs)

        return realizations

    def case_initialized(self, case):
        return self._enkf_main.getEnkfFsManager().isCaseInitialized(case)

    def get_queue_config(self):
        return self._enkf_main.get_queue_config()

    def get_number_of_iterations(self):
        return (
            self._enkf_main.analysisConfig().getAnalysisIterConfig().getNumIterations()
        )

    @property
    def have_observations(self):
        return self._enkf_main.have_observations()

    @property
    def run_path(self):
        return self._enkf_main.getModelConfig().getRunpathAsString()

    def load_from_forward_model(
        self, case: str, realisations: List[bool], iteration: int
    ) -> int:
        fs = self._enkf_main.getEnkfFsManager().getFileSystem(case)
        return self._enkf_main.loadFromForwardModel(realisations, iteration, fs)

    def get_observations(self):
        return self._enkf_main.getObservations()

    def get_impl_type_name_for_obs_key(self, key):
        return self._enkf_main.getObservations()[key].getImplementationType().name

    def get_current_fs(self):
        return self._enkf_main.getEnkfFsManager().getCurrentFileSystem()

    def get_data_key_for_obs_key(self, observation_key):
        return self._enkf_main.getObservations()[observation_key].getDataKey()

    def get_matching_wildcards(self):
        return self._enkf_main.getObservations().getMatchingKeys

    def get_observation_key(self, index):
        return self._enkf_main.getObservations()[index].getKey()

    def load_gen_data(self, case_name, key, report_step):
        return GenDataCollector.loadGenData(
            self._enkf_main, case_name, key, report_step
        )

    def load_all_summary_data(self, case_name, keys=None):
        return SummaryCollector.loadAllSummaryData(self._enkf_main, case_name, keys)

    def load_observation_data(self, case_name, keys=None):
        return SummaryObservationCollector.loadObservationData(
            self._enkf_main, case_name, keys
        )

    def create_plot_block_data_loader(self, obs_vector):
        return PlotBlockDataLoader(obs_vector)

    def select_or_create_new_case(self, case_name):
        if self.get_current_case_name() != case_name:
            fs = self._enkf_main.getEnkfFsManager().getFileSystem(case_name)
            self._enkf_main.getEnkfFsManager().switchFileSystem(fs)

    def cases(self):
        return self._enkf_main.getEnkfFsManager().getCaseList()

    def is_case_hidden(self, case):
        return self._enkf_main.getEnkfFsManager().isCaseHidden(case)

    def case_has_data(self, case):
        return self._enkf_main.getEnkfFsManager().caseHasData(case)

    def is_case_running(self, case):
        return self._enkf_main.getEnkfFsManager().isCaseRunning(case)

    def all_data_type_keys(self):
        return self._enkf_main.getKeyManager().allDataTypeKeys()

    def observation_keys(self, key):
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

    def gather_gen_kw_data(self, case, key, realization_index=None):
        """:rtype: pandas.DataFrame"""
        data = GenKwCollector.loadAllGenKwData(
            self._enkf_main, case, [key], realization_index=realization_index
        )
        if key in data:
            return data[key].to_frame().dropna()
        else:
            return DataFrame()

    def gather_summary_data(self, case, key, realization_index=None):
        """:rtype: pandas.DataFrame"""
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

    def has_refcase(self, key):
        refcase = self._enkf_main.eclConfig().getRefcase()
        return refcase is not None and key in refcase

    def refcase_data(self, key):
        refcase = self._enkf_main.eclConfig().getRefcase()

        if refcase is None or key not in refcase:
            return DataFrame()

        values = refcase.numpy_vector(key, report_only=False)
        dates = refcase.numpy_dates

        data = DataFrame(zip(dates, values), columns=["Date", key])
        data.set_index("Date", inplace=True)

        return data.iloc[1:]

    def history_data(self, key, case=None):
        if not self.is_summary_key(key):
            return DataFrame()

        # create history key
        if ":" in key:
            head, tail = key.split(":", 2)
            key = "{}H:{}".format(head, tail)
        else:
            key = "{}H".format(key)

        data = self.refcase_data(key)
        if data.empty and case is not None:
            data = self.gather_summary_data(case, key)

        return data

    def gather_gen_data_data(self, case, key, realization_index=None):
        """:rtype: pandas.DataFrame"""
        key_parts = key.split("@")
        key = key_parts[0]
        if len(key_parts) > 1:
            report_step = int(key_parts[1])
        else:
            report_step = 0

        try:
            data = GenDataCollector.loadGenData(
                self._enkf_main, case, key, report_step, realization_index
            )
        except (ValueError, KeyError):
            data = DataFrame()

        return data.dropna()  # removes all rows that has a NaN

    def is_summary_key(self, key):
        """:rtype: bool"""
        return key in self._enkf_main.getKeyManager().summaryKeys()

    def get_summary_keys(self):
        """:rtype: list of str"""
        return self._enkf_main.getKeyManager().summaryKeys()

    def is_gen_kw_key(self, key):
        """:rtype: bool"""
        return key in self._enkf_main.getKeyManager().genKwKeys()

    def gen_kw_keys(self):
        """:rtype: list of str"""
        return self._enkf_main.getKeyManager().genKwKeys()

    def is_gen_data_key(self, key):
        """:rtype: bool"""
        return key in self._enkf_main.getKeyManager().genDataKeys()

    def get_gen_data_keys(self):
        """:rtype: list of str"""
        return self._enkf_main.getKeyManager().genDataKeys()

    def gen_kw_priors(self):
        return self._enkf_main.getKeyManager().gen_kw_priors()

    def get_update_step(self):
        return self._enkf_main.getLocalConfig().getUpdatestep()

    def get_alpha(self):
        return self._enkf_main.analysisConfig().getEnkfAlpha()

    def get_std_cutoff(self):
        return self._enkf_main.analysisConfig().getStdCutoff()

    def get_workflow_job(self, name):
        if self._enkf_main.getWorkflowList().hasJob(name):
            return self._enkf_main.getWorkflowList().getJob(name)
        return None
