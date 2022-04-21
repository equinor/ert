from typing import Any, Dict

from ert_shared.models import ErtRunError, EnsembleExperiment
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig

from res.enkf.ert_run_context import ErtRunContext
from res.enkf.enkf_main import EnKFMain


class SingleTestRun(EnsembleExperiment):
    def __init__(self, simulation_arguments: Dict[str, Any], ert: EnKFMain, *_: Any):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(
            simulation_arguments,
            ert,
            local_queue_config,
        )

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        # Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        return self.runSimulations__(
            "Running single realisation test ...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
