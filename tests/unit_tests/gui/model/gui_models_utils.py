from ert.ensemble_evaluator.snapshot import ForwardModel, Snapshot
from ert.ensemble_evaluator.state import FORWARD_MODEL_STATE_FINISHED


def finish_snapshot(snapshot: Snapshot) -> Snapshot:
    snapshot._realization_states["0"].update({"status": FORWARD_MODEL_STATE_FINISHED})
    snapshot.update_forward_model(
        "0", "0", ForwardModel(status=FORWARD_MODEL_STATE_FINISHED)
    )
    return snapshot
