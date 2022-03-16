from typing import Any

from ert_shared.models import ErtRunError, EnsembleExperiment
from ert_shared.models.types import Argument
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig

from res.enkf.ert_run_context import ErtRunContext
from res.enkf.enkf_main import EnKFMain


class SingleTestRun(EnsembleExperiment):
    def __init__(self, ert: EnKFMain, *_: Any):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(ert, local_queue_config)

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        # Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(
        self, arguments: Argument, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        return self.runSimulations__(
            arguments, "Running single realisation test ...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
