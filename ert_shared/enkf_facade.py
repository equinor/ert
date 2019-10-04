from res.analysis.analysis_module import AnalysisModule
from res.analysis.enums.analysis_module_options_enum import AnalysisModuleOptionsEnum
from res.enkf import ErtRunContext


class EnkfFacade():
    """Facade to untangle ENKF Main funcs inside ERT."""

    def __init__(self, enkf_main):
        self._enkf_main = enkf_main

    def get_all_cases(self):
        fs_manager = self._enkf_main.getEnkfFsManager()
        case_list = fs_manager.getCaseList()
        return [str(case) for case in case_list if not fs_manager.isCaseHidden(case)]

    def get_active_module_name(self):
        return self._enkf_main.analysisConfig().activeModuleName()

    def get_analysis_module(self, module_name):
        return self._enkf_main.analysisConfig().getModule(module_name)

    def get_analysis_module_names(self, iterable=False):
        modules = self.get_analysis_modules(iterable)
        return [module.getName() for module in modules]

    def get_analysis_modules(self, iterable=False):
        module_names = self._enkf_main.analysisConfig().getModuleList()

        modules = []
        for module_name in module_names:
            module = self.get_analysis_module(module_name)
            module_is_iterable = module.checkOption(
                AnalysisModuleOptionsEnum.ANALYSIS_ITERABLE)

            if iterable == module_is_iterable:
                modules.append(module)

        return sorted(modules, key=AnalysisModule.getName)

    def get_case_format(self):
        return self._enkf_main.analysisConfig().getAnalysisIterConfig().getCaseFormat()

    def get_config_node(self, keyword):
        return self._enkf_main.ensembleConfig()[keyword]

    def get_current_case_name(self):
        return str(self._enkf_main.getEnkfFsManager().getCurrentFileSystem().getCaseName())

    def get_ensemble_size(self):
        return self._enkf_main.getEnsembleSize()

    def get_file_system(self, selected_case):
        return self._enkf_main.getEnkfFsManager().getFileSystem(selected_case)

    def get_forward_model(self):
        return self._enkf_main.getModelConfig().getForwardModel()

    def get_history_length(self):
        return self._enkf_main.getHistoryLength()

    def get_keylist_from_impl_type(self, impl_type):
        return self._enkf_main.ensembleConfig().getKeylistFromImplType(impl_type)

    def get_keylist_from_var_type(self, var_type):
        return self._enkf_main.ensembleConfig().getKeylistFromVarType(var_type)

    def get_node(self, keyword):
        return self._enkf_main.ensembleConfig().getNode(keyword)

    def get_node_input_format(self, keyword):
        return self.get_node(keyword).getDataModelConfig().getInputFormat()

    def get_node_output_format(self, keyword):
        return self.get_node(keyword).getDataModelConfig().getOutputFormat()

    def get_number_of_iterations(self):
        return self._enkf_main.analysisConfig().getAnalysisIterConfig().getNumIterations()

    def get_observations(self, key):
        return self._enkf_main.getObservations()[key]

    def get_observations_data_key(self, key):
        return self._enkf_main.getObservations()[key].getDataKey()

    def get_observation_keys(self, key):
        return self.get_node(key).getObservationKeys()

    def get_other_keys(self, current_key):
        key_manager = self._enkf_main.getKeyManager()
        return [k for k in key_manager.allDataTypeKeys() if k != current_key]

    def get_queue_config(self):
        return self._enkf_main.get_queue_config()

    def get_runpath_as_string(self):
        return self._enkf_main.getModelConfig().getRunpathAsString()

    def get_runpath_format(self):
        return self._enkf_main.getModelConfig().getRunpathFormat()

    def get_state_map(self, case):
        return self._enkf_main.getEnkfFsManager().getStateMapForCase(case)

    def get_typed_keylist(self, enkf_observation_impl_type):
        return self._enkf_main.getObservations().getTypedKeylist(enkf_observation_impl_type)

    def get_workflow_list(self):
        return self._enkf_main.getWorkflowList()

    def have_observations(self):
        return self._enkf_main.have_observations()

    def is_case_initialized(self, case_name):
        return self._enkf_main.getEnkfFsManager().isCaseInitialized(case_name)

    def is_case_format_set(self):
        return self._enkf_main.analysisConfig().getAnalysisIterConfig().caseFormatSet()

    def is_case_running(self, case):
        return self._enkf_main.getEnkfFsManager().isCaseRunning(case)

    def is_custom_kw_key(self, key):
        return self._enkf_main.getKeyManager().isCustomKwKey(key)

    def is_gen_data_key(self, key):
        return self._enkf_main.getKeyManager().isGenDataKey(key)

    def is_gen_kw_key(self, key):
        return self._enkf_main.getKeyManager().isGenKwKey(key)

    def is_summary_key(self, key):
        return self._enkf_main.getKeyManager().isSummaryKey(key)

    def initialize_from_existing_case(self, source_case, source_report_step, member_mask, selected_parameters):
        file_system_manager = self._enkf_main.getEnkfFsManager()
        file_system_manager.customInitializeCurrentFromExistingCase(source_case, source_report_step, member_mask,
                                                                    selected_parameters)

    def initialize_from_scratch(self, mask, selected_parameters):
        current_file_system = self._enkf_main.getEnkfFsManager().getCurrentFileSystem()
        run_context = ErtRunContext.case_init(current_file_system, mask)
        self._enkf_main.getEnkfFsManager().initializeFromScratch(
            selected_parameters, run_context)

    def load_results(self, selected_case, realisations, iteration):
        file_system = self.get_file_system(selected_case)
        return self._enkf_main.loadFromForwardModel(realisations, iteration, file_system)

    def set_number_of_iterations(self, iteration_count):
        self._enkf_main.analysisConfig().getAnalysisIterConfig(
        ).setNumIterations(iteration_count)

    def set_variable_value(self, analysis_module_name, name, value):
        analysis_module = self.get_analysis_module(analysis_module_name)
        analysis_module.setVar(name, str(value))

    def switch_file_system(self, case_name):
        file_system = self.get_file_system(case_name)
        self._enkf_main.getEnkfFsManager().switchFileSystem(file_system)

    def try_load_node(self, selected_case, node, node_id):
        file_system = self.get_file_system(selected_case)
        return node.tryLoad(file_system, node_id)
