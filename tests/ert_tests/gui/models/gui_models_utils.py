from ert.ensemble_evaluator.snapshot import (
    Job,
    PartialSnapshot,
    Realization,
)
from ert.ensemble_evaluator.state import JOB_STATE_FINISHED


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial.update_real("0", Realization(status=JOB_STATE_FINISHED))
    partial.update_job("0", "0", "0", Job(status=JOB_STATE_FINISHED))
    return partial
