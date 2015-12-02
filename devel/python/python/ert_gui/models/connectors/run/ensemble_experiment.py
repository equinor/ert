from ert_gui.models.connectors.run import ActiveRealizationsModel, BaseRunModel
from ert_gui.models.mixins.run_model import ErtRunError


class EnsembleExperiment(BaseRunModel):

    def __init__(self):
        super(EnsembleExperiment, self).__init__("Ensemble Experiment")

    def runSimulations(self):
        self.setPhase(0, "Running simulations...", indeterminate=False)
        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()
        
        if self.ert().getEnkfSimulationRunner().isHookPreSimulation():
            self.ert().getEnkfSimulationRunner().runHookWorkflow()
            
        success = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(active_realization_mask)

        if not success:
            raise ErtRunError("Simulation failed!")

        self.setPhaseName("Post processing...", indeterminate=True)
        
        if self.ert().getEnkfSimulationRunner().isHookPostSimulation():
            self.ert().getEnkfSimulationRunner().runHookWorkflow()
        
        self.ert().getEnkfSimulationRunner().runPostHookWorkflow()

        self.setPhase(1, "Simulations completed.") # done...



