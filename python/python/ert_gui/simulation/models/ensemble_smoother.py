from res.enkf.enums import EnkfInitModeEnum
from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError


class EnsembleSmoother(BaseRunModel):

    def __init__(self, queue_config):
        super(EnsembleSmoother, self).__init__("Ensemble Smoother", queue_config , phase_count=2)

    def setAnalysisModule(self, module_name):
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)


    def runSimulations(self, job_queue, run_context):
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(job_queue, run_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhaseName("Analyzing...")

        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_UPDATE )
        es_update = self.ert().getESUpdate( ) 
        success = es_update.smootherUpdate( run_context )
        if not success:
            raise ErtRunError("Analysis of simulation failed!")
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_UPDATE )

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem(target_fs)
        
        self.setPhaseName("Pre processing...")
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, 1)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(job_queue, active_realization_mask, EnkfInitModeEnum.INIT_NONE, 1)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhase(2, "Simulations completed.")
