import copy
import logging
from typing import Dict, List, Tuple

from .job_queue_node import JobQueueNode
from .job_status import JobStatus

logger = logging.getLogger(__name__)


class QueueDiffer:
    def __init__(self) -> None:
        self._qindex_to_iens: Dict[int, int] = {}
        self._state: List[JobStatus] = []

    def add_state(self, queue_index: int, iens: int, state: JobStatus) -> None:
        self._qindex_to_iens[queue_index] = iens
        self._state.append(state)

    def get_old_and_new_state(
        self,
        job_list: List[JobQueueNode],
    ) -> Tuple[List[JobStatus], List[JobStatus]]:
        """Calculate a new state, do not transition, return both old and new state."""
        new_state = [job.queue_status.value for job in job_list]
        old_state = copy.copy(self._state)
        return old_state, new_state

    def transition_to_new_state(self, new_state: List[JobStatus]) -> None:
        self._state = new_state

    def transition(
        self,
        job_list: List[JobQueueNode],
    ) -> Tuple[List[JobStatus], List[JobStatus]]:
        """Transition to a new state, return both old and new state."""
        new_state = [job.queue_status.value for job in job_list]
        old_state = copy.copy(self._state)
        self._state = new_state
        return old_state, new_state

    def diff_states(
        self,
        old_state: List[JobStatus],
        new_state: List[JobStatus],
    ) -> Dict[int, str]:
        """Return the diff between old_state and new_state."""
        changes = {}

        diff = [s[0] == s[1] for s in zip(old_state, new_state)]
        if len(diff) > 0:
            for q_index, equal in enumerate(diff):
                if not equal:
                    st = str(JobStatus(new_state[q_index]))
                    changes[self._qindex_to_iens[q_index]] = st
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

    def qindex_to_iens(self, queue_index: int) -> int:
        return self._qindex_to_iens[queue_index]
