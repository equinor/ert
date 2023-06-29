from ert.ensemble_evaluator.snapshot import Job, PartialSnapshot
from ert.ensemble_evaluator.state import JOB_STATE_FINISHED


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial._realization_states["0"].update({"status": JOB_STATE_FINISHED})
    partial.update_job("0", "0", "0", Job(status=JOB_STATE_FINISHED))
    return partial
