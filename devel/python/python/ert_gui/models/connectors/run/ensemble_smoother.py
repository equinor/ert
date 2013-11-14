import time
from ert.enkf.enums import EnkfInitModeEnum
from ert_gui.models import ErtConnector
from ert_gui.models.connectors.run import ActiveRealizationsModel, TargetCaseModel, AnalysisModuleModel
from ert_gui.models.mixins import RunModelMixin


class EnsembleSmoother(ErtConnector, RunModelMixin):

    def __init__(self):
        super(EnsembleSmoother, self).__init__(phase_count=2)

    def startSimulations(self):
        module_name = AnalysisModuleModel().getCurrentChoice()
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            self.runFailed("Unable to load analysis module '%s'!" % module_name)
            return

        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()

        self.ert().getEnkfSimulationRunner().runSimpleStep(active_realization_mask, EnkfInitModeEnum.INIT_CONDITIONAL)

        target_case_name = TargetCaseModel().getValue()
        target_fs = self.ert().getEnkfFsManager().mountAlternativeFileSystem(target_case_name, read_only=False, create=True)

        success = self.ert().getEnkfSimulationRunner().smootherUpdate(target_fs)

        if not success:
            self.runFailed("Analysis of simulation failed!")
            return

        self.setPhase(1)

        self.ert().getEnkfFsManager().switchFileSystem(target_fs)
        self.ert().getEnkfSimulationRunner().runSimpleStep(active_realization_mask, EnkfInitModeEnum.INIT_NONE)

        self.setPhase(2)


    def killAllSimulations(self):
        job_queue = self.ert().siteConfig().getJobQueue()
        job_queue.killAllJobs()

    def __str__(self):
        return "Ensemble Smoother"



