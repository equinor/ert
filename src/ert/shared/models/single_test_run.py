from typing import Any, Dict
from uuid import UUID

from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import EnsembleExperiment, ErtRunError
from ert.storage import StorageAccessor


class SingleTestRun(EnsembleExperiment):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        storage: StorageAccessor,
        id_: UUID,
        *_: Any
    ):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(simulation_arguments, ert, storage, local_queue_config, id_)

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        # Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        return self.runSimulations__(
            "Running single realisation test ...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
