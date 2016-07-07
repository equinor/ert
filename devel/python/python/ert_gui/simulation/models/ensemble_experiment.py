from ert_gui.models.connectors.run import ActiveRealizationsModel
from ert.enkf.enums import HookRuntime
from ert.enkf import ErtLog
from ert_gui.simulation.models import BaseRunModel, ErtRunError


class EnsembleExperiment(BaseRunModel):

    def __init__(self):
        super(EnsembleExperiment, self).__init__("Ensemble Experiment")

    def runSimulations(self, arguments):
        self.setPhase(0, "Running simulations...", indeterminate=False)
        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, 0)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running ensemble experiment...", indeterminate=False)

        success = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(active_realization_mask)

        if not success:
            raise ErtRunError("Simulation or loading of results failed! \n"
                              "Check ERT log file '%s' or simulation folder for details." % ErtLog.getFilename())

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhase(1, "Simulations completed.") # done...



