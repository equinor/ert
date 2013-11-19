import time
from ert_gui.models.connectors.run import NumberOfIterationsModel, ActiveRealizationsModel, IteratedAnalysisModuleModel, BaseRunModel
from ert_gui.models.mixins import ErtRunError


class IteratedEnsembleSmoother(BaseRunModel):

    def __init__(self):
        super(IteratedEnsembleSmoother, self).__init__(name="Iterated Ensemble Smoother", phase_count=2)

    def setAnalysisModule(self):
        module_name = IteratedAnalysisModuleModel().getCurrentChoice()
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)


    def runSimulations(self):
        iteration_count = NumberOfIterationsModel().getValue()
        phase_count = iteration_count
        self.setPhaseCount(phase_count)

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setAnalysisModule()

        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()

        tts = 2
        for phase in range(self.phaseCount()):

            # if phase == 3:
            #     raise ErtRunError("Phase #%d failed!" % (phase + 1))

            self.setPhase(phase, "Running iteration %d of %d simulation iterations..." % (phase + 1, phase_count), indeterminate=False)
            time.sleep(tts)

            self.setPhaseName("Analyzing...", indeterminate=True)
            time.sleep(tts)

        self.setPhase(phase_count, "Simulations completed.")


