import copy
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

# from .job_queue_node import JobQueueNode
from .job_status import JobStatus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ert.job_queue import ExecutableRealization

# For refactoring, we need to figure out if the JobStatus of a realization should
# be present in the ExecutableRealization aka JobQueueNode object, or should it be a property
# of the driver, or maintained in a dict in JobQueue mapping from iens to JobStatus

# Do we need this qindex number really?


class QueueDiffer:
    def __init__(self) -> None:
        self._qindex_to_iens: Dict[int, int] = {}
        self._state: List[JobStatus] = []

    def add_state(self, queue_index: int, iens: int, state: JobStatus) -> None:
        self._qindex_to_iens[queue_index] = iens
        self._state.append(state)

    def transition(
        self,
        job_list: List["ExecutableRealization"],
    ) -> Tuple[List[JobStatus], List[JobStatus]]:
        """Transition to a new state, return both old and new state."""
        new_state = [job.queue_status.value for job in job_list]
        old_state = copy.copy(self._state)
        self._state = new_state
        return old_state, new_state

    def diff_states(
        self,
        old_state: Dict["ExecutableRealization", JobStatus],
        new_state: Dict["ExecutableRealization", JobStatus],
    ) -> Dict[int, str]:
        """Return the diff between old_state and new_state.

        Job statuses are returned as string for json serializability
        """
        changes = {}

        if not set(old_state.keys()).issubset(new_state.keys()):
            print(" <queue_differ> The dictionary of states should not shrink")
            print(f" <queue_differ> {old_state.keys()=} {new_state.keys()=}")

        for job in new_state:
            if job in old_state:
                if old_state[job] != new_state[job]:
                    changes[job.run_arg.iens] = str(new_state[job])
            else:
                changes[job.run_arg.iens] = str(new_state[job])
        return changes

    def snapshot(self) -> Dict[int, str]:
        """Return the whole state, or None if there was no snapshot."""
        snapshot = {}
        for q_index, state_val in enumerate(self._state):
            st = str(JobStatus(state_val))
            try:
                snapshot[self._qindex_to_iens[q_index]] = st
            except KeyError as e:
                logger.debug(f"differ could produce no snapshot due to {e}")
                raise e from None
        return snapshot
