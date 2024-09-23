from ert.ensemble_evaluator.snapshot import EnsembleSnapshot, FMStepSnapshot
from ert.ensemble_evaluator.state import FORWARD_MODEL_STATE_FINISHED


def finish_snapshot(snapshot: EnsembleSnapshot) -> EnsembleSnapshot:
    snapshot._realization_snapshots["0"].update(
        {"status": FORWARD_MODEL_STATE_FINISHED}
    )
    snapshot.update_fm_step(
        "0", "0", FMStepSnapshot(status=FORWARD_MODEL_STATE_FINISHED)
    )
    return snapshot
