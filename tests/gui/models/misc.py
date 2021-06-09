from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Realization, Job
from ert_shared.status.entity.state import JOB_STATE_FINISHED


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial.update_real("0", Realization(status=JOB_STATE_FINISHED))
    partial.update_job("0", "0", "0", Job(status=JOB_STATE_FINISHED))
    return partial
