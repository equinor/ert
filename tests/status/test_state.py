from res.job_queue import JobStatusType
from ert_shared.tracker.state import create_states


def check_for_unused_enums():
    states = create_states()
    for enum in JobStatusType.enums():
        # The status check routines can return this status; if e.g. the bjobs call fails,
        # but a job will never get this status.
        if enum == JobStatusType.JOB_QUEUE_STATUS_FAILURE:
            continue

        used = False
        for state in states:
            if enum in state.state:
                used = True

        if not used:
            raise AssertionError("Enum identifier '%s' not used!" % enum)
