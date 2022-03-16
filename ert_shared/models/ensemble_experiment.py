from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext, EnkfSimulationRunner

from ert_shared.models import BaseRunModel
from ert_shared.models.types import Argument
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig


class EnsembleExperiment(BaseRunModel):
    def __init__(self, ert: EnKFMain, queue_config: QueueConfig):
        super().__init__(ert, queue_config)

    def runSimulations__(
        self,
        arguments: Argument,
        run_msg: str,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> ErtRunContext:

        run_context = self.create_context(arguments)

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, self.ert())

        self.setPhaseName(run_msg, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        num_successful_realizations += arguments.get("prev_successful_realizations", 0)
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, self.ert())

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(1, "Simulations completed.")  # done...

        return run_context

    def runSimulations(
        self, arguments: Argument, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        return self.runSimulations__(
            arguments, "Running ensemble experiment...", evaluator_server_config
        )

    def create_context(self, arguments: Argument) -> ErtRunContext:
        fs_manager = self.ert().getEnkfFsManager()
        result_fs = fs_manager.getCurrentFileSystem()

        model_config = self.ert().getModelConfig()
        runpath_fmt = model_config.getRunpathFormat()
        jobname_fmt = model_config.getJobnameFormat()
        subst_list = self.ert().getDataKW()
        itr = arguments.get("iter_num", 0)
        mask = arguments["active_realizations"]

        run_context = ErtRunContext.ensemble_experiment(
            result_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr
        )

        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble Experiment"
