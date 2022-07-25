import pickle
import sys
import time
from concurrent import futures
from typing import Dict, Optional, Union, Any

import ert
from ert_shared.cli.monitor import Monitor
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
from ert.ensemble_evaluator.identifiers import (
    EVTYPE_EE_TERMINATED,
)


class ERT3RunModel:
    # pylint: disable=too-many-instance-attributes
    # some attributes might be encapsulated in the future
    def __init__(self, phase_count: int = 1):
        self._phase: int = 0
        self._phase_count: int = phase_count
        self._phase_name: str = "Starting..."
        self._job_start_time: int = 0
        self._job_stop_time: int = 0
        self._fail_message: str = ""
        self._failed: bool = False
        self._simulation_arguments: Dict[str, Any] = {}
        self.support_restart: bool = False

    def teardown_context(self) -> None:
        return None

    def isFinished(self) -> bool:
        return self._phase_count == self._phase or self.hasRunFailed()

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

    def restart(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self._failed = False
        self._phase = 0

    def stop_time(self) -> int:
        return self._job_stop_time

    def start_time(self) -> int:
        return self._job_start_time

    def get_runtime(self) -> Union[int, float]:
        if self.stop_time() < self.start_time():
            return time.time() - self.start_time()
        else:
            return self.stop_time() - self.start_time()

    def has_failed_realizations(self) -> bool:
        return False


def _run(
    ensemble_evaluator: EnsembleEvaluator,
    run_model: ERT3RunModel,
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
    ensemble: ert.ensemble_evaluator.Ensemble,
    custom_port_range: Optional[range] = None,
    use_gui: bool = False,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    config = EvaluatorServerConfig(custom_port_range=custom_port_range)

    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    if use_gui:
        # pylint: disable=import-outside-toplevel
        from ert.gui.simulation.run_dialog import run_monitoring_ert3

        return run_monitoring_ert3(ee, ERT3RunModel())

    result: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {}
    with futures.ThreadPoolExecutor() as executor:
        run_model = ERT3RunModel()
        tracker = ert.ensemble_evaluator.EvaluatorTracker(
            run_model,
            config.get_connection_info(),
        )
        monitor = Monitor(out=sys.stderr, color_always=False)  # type: ignore
        future = executor.submit(_run, ee, run_model)
        monitor.monitor(tracker)  # type: ignore
        result = future.result()
    return result
