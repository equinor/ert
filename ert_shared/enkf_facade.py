from res.analysis.analysis_module import AnalysisModule
from res.analysis.enums.analysis_module_options_enum import AnalysisModuleOptionsEnum

class EnkfFacade():
    """Facade to untangle ENKF Main funcs inside ERT."""

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
