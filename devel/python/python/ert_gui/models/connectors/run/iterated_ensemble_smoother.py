from ert.enkf.enums import EnkfInitModeEnum, EnkfStateType
from ert_gui.models.connectors.run import NumberOfIterationsModel, ActiveRealizationsModel, IteratedAnalysisModuleModel, BaseRunModel
from ert_gui.models.connectors.run.target_case_format_model import TargetCaseFormatModel
from ert_gui.models.mixins import ErtRunError


class IteratedEnsembleSmoother(BaseRunModel):

    def __init__(self):
        super(IteratedEnsembleSmoother, self).__init__(name="Iterated Ensemble Smoother", phase_count=2)

    def setAnalysisModule(self):
        module_name = IteratedAnalysisModuleModel().getCurrentChoice()
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)

        return self.ert().analysisConfig().getModule(module_name)


    def runAndPostProcess(self, active_realization_mask, phase, phase_count, mode):
        self.setPhase(phase, "Running iteration %d of %d simulation iterations..." % (phase, phase_count - 1), indeterminate=False)

        success = self.ert().getEnkfSimulationRunner().runSimpleStep(active_realization_mask, mode, phase)

        if not success:
            min_realization_count = self.ert().analysisConfig().getMinRealisations()
            success_count = active_realization_mask.count()

            if min_realization_count > success_count:
                raise ErtRunError("Simulation failed! Number of successful realizations less than MIN_REALIZATIONS %d < %d" % (success_count, min_realization_count))
            elif success_count == 0:
                raise ErtRunError("Simulation failed! All realizations failed!")
            #ignore and continue

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runPostWorkflow()


    def createTargetCaseFileSystem(self, phase):
        target_case_format = TargetCaseFormatModel().getValue()
        target_fs = self.ert().getEnkfFsManager().getFileSystem(target_case_format % phase)
        return target_fs


    def analyzeStep(self, target_fs):
        self.setPhaseName("Analyzing...", indeterminate=True)
        success = self.ert().getEnkfSimulationRunner().smootherUpdate(target_fs)

        if not success:
            raise ErtRunError("Analysis of simulation failed!")


    def runSimulations(self):
        iteration_count = NumberOfIterationsModel().getValue()
        phase_count = iteration_count + 1
        self.setPhaseCount(phase_count)

        analysis_module = self.setAnalysisModule()
        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()

        source_fs = self.ert().getEnkfFsManager().getCurrentFileSystem()
        initial_fs = self.createTargetCaseFileSystem(0)

        if not source_fs == initial_fs:
            self.ert().getEnkfFsManager().switchFileSystem(initial_fs)
            self.ert().getEnkfFsManager().initializeCurrentCaseFromExisting(source_fs, 0, EnkfStateType.ANALYZED)

        analysis_module.setVar("ITER", str(0))
        self.runAndPostProcess(active_realization_mask, 0, phase_count, EnkfInitModeEnum.INIT_CONDITIONAL)
        target_case_format = TargetCaseFormatModel().getValue()
        self.ert().analysisConfig().getAnalysisIterConfig().setCaseFormat( target_case_format )

        for phase in range(1, phase_count):
            target_fs = self.createTargetCaseFileSystem(phase)

            self.analyzeStep(target_fs)

            self.ert().getEnkfFsManager().switchFileSystem(target_fs)

            analysis_module.setVar("ITER", str(phase))
            self.runAndPostProcess(active_realization_mask, phase, phase_count, EnkfInitModeEnum.INIT_NONE)

        self.setPhase(phase_count, "Simulations completed.")
