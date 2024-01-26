from ert.ensemble_evaluator.snapshot import ForwardModel, PartialSnapshot
from ert.ensemble_evaluator.state import FORWARD_MODEL_STATE_FINISHED


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial._realization_states["0"].update({"status": FORWARD_MODEL_STATE_FINISHED})
    partial.update_forward_model(
        "0", "0", ForwardModel(status=FORWARD_MODEL_STATE_FINISHED)
    )
    return partial
