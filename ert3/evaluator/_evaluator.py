import pickle
import time
from typing import Dict, Optional, Union, Any
from concurrent import futures
import sys

import ert
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models.base_run_model import BaseRunModel
from ert_shared.status.tracker.evaluator import EvaluatorTracker
from ert_shared.cli.monitor import Monitor
from ert_shared.ensemble_evaluator.ensemble.base import Ensemble
from ert_shared.ensemble_evaluator.entity.identifiers import (
    EVTYPE_EE_TERMINATED,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator

from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)


class DummyBaseRunModel(BaseRunModel):
    def __init__(self, phase_count: int = 1):
        self._phase: int = 0
        self._phase_count: int = phase_count
        self._phase_name: str = "Starting..."
        self._job_start_time: int = 0
        self._job_stop_time: int = 0
        self._indeterminate: bool = False
        self._fail_message: str = ""
        self._failed: bool = False

    def create_context(self, arguments) -> Any:
        return None

    def runSimulations(self, arguments):
        return self.create_context(arguments)

    def isFinished(self) -> bool:
        return self._phase_count == self._phase or self.hasRunFailed()

    def teardown_context(self) -> None:
        return None

    def hasRunFailed(self) -> bool:
        return self._failed

    def getFailMessage(self) -> str:
        return self._fail_message

    def getPhaseName(self) -> str:
        return self._phase_name

    def currentPhase(self) -> int:
        return self._phase

    def phaseCount(self) -> int:
        return self._phase_count

    def setPhase(
        self, phase: int, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self._phase_name = phase_name
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                "Phase must be an integer between (inclusive) 0 and {self._phase_count}"
            )

        if phase == 0:
            self._job_start_time = int(time.time())

        if phase == self._phase_count:
            self._job_stop_time = int(time.time())

        self._phase = phase

    def isIndeterminate(self) -> bool:
        return False

    def setIndeterminate(self, indeterminate: Union[bool, None]) -> None:
        raise NotImplementedError("This function is not in use in ert3")


def _run(
    ensemble_evaluator: EnsembleEvaluator,
    run_model: DummyBaseRunModel,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    result: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {}
    with ensemble_evaluator.run() as monitor:
        run_model.setPhase(0, "Running simulations...", indeterminate=False)
        for event in monitor.track():
            if isinstance(event.data, dict) and event.data.get("status") in [
                ENSEMBLE_STATE_STOPPED,
                ENSEMBLE_STATE_FAILED,
            ]:
                monitor.signal_done()
                if event.data.get("status") == ENSEMBLE_STATE_FAILED:
                    run_model._failed = True
                    run_model._fail_message = "Ensemble evaluation failed"
                    raise RuntimeError("Ensemble evaluation failed")
            if event["type"] == EVTYPE_EE_TERMINATED and isinstance(event.data, bytes):
                run_model.setPhase(1, "Simulations completed.")
                result = pickle.loads(event.data)

    return result


def evaluate(
    ensemble: Ensemble, cli: bool = True
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    config = EvaluatorServerConfig()

    run_model = DummyBaseRunModel(1)
    if cli:
        tracker = EvaluatorTracker(
            run_model,
            config.host,
            config.port,
            1,
            0,
            token=config.token,
            cert=config.cert,
        )
        monitor = Monitor(out=sys.stderr, color_always=False)  # type: ignore
    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)

    executor = futures.ThreadPoolExecutor()
    future = executor.submit(_run, ee, run_model)
    if cli:
        try:
            monitor.monitor(tracker)  # type: ignore
        except (SystemExit, KeyboardInterrupt):
            tracker.request_termination()  # type: ignore
    result: Dict[int, Dict[str, ert.data.RecordTransmitter]] = future.result()
    return result
