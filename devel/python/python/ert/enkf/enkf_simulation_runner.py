from ert.cwrap import CWrapper, BaseCClass
from ert.enkf import ENKF_LIB, EnkfStateType
from ert.util import BoolVector


class EnkfSimulationRunner(BaseCClass):

    def __init__(self, enkf_main):
        assert isinstance(enkf_main, BaseCClass)
        super(EnkfSimulationRunner, self).__init__(enkf_main.from_param(enkf_main).value, parent=enkf_main, is_reference=True)


    def runEnsembleExperiment(self, active_realization_mask):
        assert isinstance(active_realization_mask, BoolVector)
        EnkfSimulationRunner.cNamespace().run_exp(self, active_realization_mask, True, 0, 0, EnkfStateType.ANALYZED, True)




    # def run(self, boolPtr, init_step_parameter, simFrom, state, mode):
    #     #{"ENKF_ASSIMILATION" : 1, "ENSEMBLE_EXPERIMENT" : 2, "ENSEMBLE_PREDICTION" : 3, "INIT_ONLY" : 4, "SMOOTHER" : 5}
    #     if mode == 1:
    #         EnKFMain.cNamespace().run_assimilation(self, boolPtr, init_step_parameter, simFrom, state)
    #
    #     if mode == 2:
    #         EnKFMain.cNamespace().run_exp(self, boolPtr, True, init_step_parameter, simFrom, state, True)
    #
    #     if mode == 4:
    #         EnKFMain.cNamespace().run_exp(self, boolPtr, False, init_step_parameter, simFrom, state , True)
    #
    #     if mode == 5:
    #         EnKFMain.cNamespace().run_smoother(self, "AUTOSMOOTHER", True)

    # def runIteratedEnsembleSmoother(self, last_report_step):
    #     #warn: Remember to select correct analysis module RML
    #     EnKFMain.cNamespace().run_iterated_ensemble_smoother(self, last_report_step)
    #
    # def runOneMoreIteration(self, last_report_step):
    #     #warn: need some way of validating that the case has run
    #     EnKFMain.cNamespace().run_one_more_iteration(self, last_report_step)

cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("enkf_simulation_runner", EnkfSimulationRunner)

EnkfSimulationRunner.cNamespace().run_exp           = cwrapper.prototype("void enkf_main_run_exp(enkf_simulation_runner, bool_vector, bool, int, int, enkf_state_type_enum, bool)")
EnkfSimulationRunner.cNamespace().run_assimilation  = cwrapper.prototype("void enkf_main_run_assimilation(enkf_simulation_runner, bool_vector, int, int, int)")
EnkfSimulationRunner.cNamespace().run_smoother      = cwrapper.prototype("void enkf_main_run_smoother(enkf_simulation_runner, char*, bool)")