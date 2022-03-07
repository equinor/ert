from ert_shared.models import ErtRunError, EnsembleExperiment


class SingleTestRun(EnsembleExperiment):
    def __init__(self, ert, *_):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(ert, local_queue_config)

    def checkHaveSufficientRealizations(self, num_successful_realizations):
        # Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(self, arguments, evaluator_server_config):
        return self.runSimulations__(
            arguments, "Running single realisation test ...", evaluator_server_config
        )

    @classmethod
    def name(cls):
        return "Single realization test-run"
