import pickle
from typing import Dict

import ert
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.ensemble.base import Ensemble
from ert_shared.ensemble_evaluator.entity.identifiers import EVTYPE_EE_TERMINATED
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator

from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
)


def _run(
    ensemble_evaluator: EnsembleEvaluator,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    result: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {}
    with ensemble_evaluator.run() as monitor:
        realization_ids = set()
        realizations_completed = set()
        for event in monitor.track():
            if isinstance(event.data, dict) and event.data.get("status") in [
                ENSEMBLE_STATE_STOPPED,
                ENSEMBLE_STATE_FAILED,
            ]:
                monitor.signal_done()
                if event.data.get("status") == ENSEMBLE_STATE_FAILED:
                    raise RuntimeError("Ensemble evaluation failed")
            if event["type"] == EVTYPE_EE_TERMINATED and isinstance(event.data, bytes):
                result = pickle.loads(event.data)
            if isinstance(event.data, dict) and "reals" in event.data:
                for real_id in event.data["reals"]:
                    realization_ids.add(real_id)
                    real_status = event.data["reals"][real_id].get("status")
                    if (
                        real_status == REALIZATION_STATE_FINISHED
                        and real_id not in realizations_completed
                    ):
                        realizations_completed.add(real_id)
                        print(
                            f"Realization {real_id} completed successfully"
                            f" ({len(realizations_completed)}/{len(realization_ids)})"
                        )

    return result


def evaluate(
    ensemble: Ensemble,
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    config = EvaluatorServerConfig()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    result = _run(ee)
    return result
