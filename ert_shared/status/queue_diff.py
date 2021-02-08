import copy
import typing
from res.job_queue import JobQueue
from res.job_queue.job_status_type_enum import JobStatusType
import logging

logger = logging.getLogger(__name__)


class InconsistentIndicesError(Exception):
    pass


class QueueDiff:
    def __init__(self, queue: JobQueue) -> None:
        self._queue = queue
        self._initialize_state()

    def _initialize_state(self):
        self._qindex_to_iens = {
            q_index: q_node.callback_arguments[0].iens
            for q_index, q_node in enumerate(self._queue.job_list)
        }
        self._state = [q_node.status.value for q_node in self._queue.job_list]

    def iens_from_queue_index(self, queue_index: int) -> int:
        return self._qindex_to_iens[queue_index]

    def transition(
        self,
    ) -> typing.Tuple[typing.List[JobStatusType], typing.List[JobStatusType]]:
        """Transition to a new state, return both old and new state."""
        new_state = [job.status.value for job in self._queue.job_list]
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
                    try:
                        changes[self._qindex_to_iens[q_index]] = st
                    except KeyError as e:
                        raise InconsistentIndicesError(
                            f"Accessing the qindex to iens map failed with {e}, job_list has probably changed between transitions"
                        )
        return changes

    def changes_after_transition(self) -> typing.Dict[int, str]:
        old_state, new_state = self.transition()
        try:
            return self.diff_states(old_state, new_state)
        except InconsistentIndicesError as e:
            logger.debug(f"Got {e}, re-initializing state")
            self._initialize_state()
            raise

    def snapshot(self) -> typing.Dict[int, str]:
        """Return the whole state"""
        snapshot = {}
        for q_index, state_val in enumerate(self._state):
            st = str(JobStatusType(state_val))
            snapshot[self._qindex_to_iens[q_index]] = st
        return snapshot

    def queue(self) -> JobQueue:
        return self._queue
