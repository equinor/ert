from res.analysis.analysis_module import AnalysisModule
from res.analysis.enums.analysis_module_options_enum import \
    AnalysisModuleOptionsEnum
from res.enkf.export import (GenDataCollector, SummaryCollector,
                             SummaryObservationCollector)
from res.enkf.plot_data import PlotBlockDataLoader


class LibresFacade(object):
    """Facade for libres inside ERT."""

    def __init__(self, enkf_main):
        self._enkf_main = enkf_main

    def get_analysis_module_names(self, iterable=False):
        modules = self.get_analysis_modules(iterable)
        return [module.getName() for module in modules]

    def get_analysis_modules(self, iterable=False):
        module_names = self._enkf_main.analysisConfig().getModuleList()

        modules = []
        for module_name in module_names:
            module = self._enkf_main.analysisConfig().getModule(module_name)
            module_is_iterable = module.checkOption(AnalysisModuleOptionsEnum.ANALYSIS_ITERABLE)

            if iterable == module_is_iterable:
                modules.append(module)

        return sorted(modules, key=AnalysisModule.getName)

    def get_ensemble_size(self):
        return self._enkf_main.getEnsembleSize()

    def get_current_case_name(self):
        return str(self._enkf_main.getEnkfFsManager().getCurrentFileSystem().getCaseName())

    def get_queue_config(self):
        return self._enkf_main.get_queue_config()

    def get_number_of_iterations(self):
        return self._enkf_main.analysisConfig().getAnalysisIterConfig().getNumIterations()

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
        return SummaryCollector.loadAllSummaryData(
            self._enkf_main, case_name, keys
        )

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
