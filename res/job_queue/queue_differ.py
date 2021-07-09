import copy
import logging
import typing

from res.job_queue.job_status_type_enum import JobStatusType

logger = logging.getLogger(__name__)


class QueueDiffer:
    def __init__(self) -> None:
        self._qindex_to_iens: typing.Dict[int, int] = {}
        self._state: typing.List[JobStatusType] = []

    def add_state(self, queue_index, iens, state):
        self._qindex_to_iens[queue_index] = iens
        self._state.append(state)

    def transition(
        self,
        job_list,
    ) -> typing.Tuple[typing.List[JobStatusType], typing.List[JobStatusType]]:
        """Transition to a new state, return both old and new state."""
        new_state = [job.status.value for job in job_list]
        old_state = copy.copy(self._state)
        self._state = new_state
        return old_state, new_state

    def diff_states(
        self,
        old_state: typing.List[JobStatusType],
        new_state: typing.List[JobStatusType],
    ) -> typing.Dict[int, str]:
        """Return the diff between old_state and new_state."""
        changes = {}

        diff = list(map(lambda s: s[0] == s[1], zip(old_state, new_state)))
        if len(diff) > 0:
            for q_index, equal in enumerate(diff):
                if not equal:
                    st = str(JobStatusType(new_state[q_index]))
                    changes[self._qindex_to_iens[q_index]] = st
        return changes

    def snapshot(self) -> typing.Optional[typing.Dict[int, str]]:
        """Return the whole state, or None if there was no snapshot."""
        snapshot = {}
        for q_index, state_val in enumerate(self._state):
            st = str(JobStatusType(state_val))
            try:
                snapshot[self._qindex_to_iens[q_index]] = st
            except KeyError as e:
                logger.debug(f"differ could produce no snapshot due to {e}")
                return None
        return snapshot

    def qindex_to_iens(self, queue_index):
        return self._qindex_to_iens[queue_index]
