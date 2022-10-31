import asyncio
from typing import Any, Dict, Union

from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import EnsembleExperiment, ErtRunError


class SingleTestRun(EnsembleExperiment):
    def __init__(
        self, simulation_arguments: Dict[str, Any], ert: EnKFMain, id_: str, *_: Any
    ):
        local_queue_config = ert.get_queue_config().create_local_copy()
        super().__init__(id_, simulation_arguments, ert, local_queue_config)

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        # Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(self, evaluator_server_config: EvaluatorServerConfig) -> Union[None, RunContext]:  # type: ignore
        return asyncio.run(
            self.run(evaluator_server_config, "single realisation test"), debug=True
        )

    async def run(
        self, evaluator_server_config: "EvaluatorServerConfig", model_name: str
    ) -> Union[None, RunContext]:
        return await super().run(evaluator_server_config, model_name)

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
