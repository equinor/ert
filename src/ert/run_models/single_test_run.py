from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from ert.run_models import EnsembleExperiment, ErtRunError

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.ensemble_evaluator import EvaluatorServerConfig
    from ert.run_context import RunContext
    from ert.run_models.run_arguments import SingleTestRunArguments
    from ert.storage import StorageAccessor


class SingleTestRun(EnsembleExperiment):
    def __init__(
        self,
        simulation_arguments: SingleTestRunArguments,
        ert: EnKFMain,
        storage: StorageAccessor,
        id_: UUID,
    ):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(simulation_arguments, ert, storage, local_queue_config, id_)

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        # Should only have one successful realization
        if num_successful_realizations != 1:
            raise ErtRunError("Experiment failed!")

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        return self.runSimulations__(
            "Running single realisation test ...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
