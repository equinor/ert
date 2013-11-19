from ert_gui.models.connectors.run import ActiveRealizationsModel, BaseRunModel


class EnsembleExperiment(BaseRunModel):

    def __init__(self):
        super(EnsembleExperiment, self).__init__("Ensemble Experiment")

    def runSimulations(self):
        self.setPhase(0, "Running simulations...", indeterminate=False)
        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()
        self.ert().getEnkfSimulationRunner().runEnsembleExperiment(active_realization_mask)
        self.setPhase(1, "Simulations completed.") # done...



