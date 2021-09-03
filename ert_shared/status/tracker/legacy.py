"""Track evaluations that does not use the Ensemble Evaluator.
Tracking happens cross-iteration, which means there's complexity pertaining to
job queues and run_context being created, and then abandoned by the experiment.

A FullSnapshotEvent will be emitted at the beginning of each iteration. Within
the life-span of an iteration, zero or more SnapshotUpdateEvent will be
emitted. A final EndEvent is emitted when the experiment is over.
"""

import logging
import time
import typing

from ert_shared.ensemble_evaluator.entity.identifiers import (
    CURRENT_MEMORY_USAGE,
    MAX_MEMORY_USAGE,
)
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    Job,
    Realization,
    SnapshotDict,
    Step,
)
from ert_shared.models.base_run_model import BaseRunModel
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_START,
    queue_status_to_real_state,
)
from ert_shared.status.utils import tracker_progress
from res.enkf.ert_run_context import ErtRunContext
from res.job_queue.job_status_type_enum import JobStatusType
from res.job_queue.queue import JobQueue

logger = logging.getLogger(__name__)

_THE_EMPTY_DETAILED_PROGRESS = ({}, -1)


_JOB_LEGACY_STATUS_MAP = {
    "Waiting": JOB_STATE_START,
    "Success": JOB_STATE_FINISHED,
    "Failure": JOB_STATE_FAILURE,
}


def _map_job_state(legacy_state: str) -> str:
    if legacy_state in _JOB_LEGACY_STATUS_MAP:
        return _JOB_LEGACY_STATUS_MAP[legacy_state]
    return legacy_state


class LegacyTracker:
    def __init__(
        self,
        model: BaseRunModel,
        general_interval: int,
        detailed_interval: int,
    ) -> None:
        self._model = model

        self._iter_snapshot = {}
        self._iter_queue = {}

        self._general_interval = general_interval
        self._detailed_interval = detailed_interval

    def track(self) -> None:
        tick = 0
        current_iter = -1
        while not self.is_finished():
            time.sleep(1)
            run_context = self._model.get_run_context()

            if run_context is None:
                logger.debug(f"no run_context at tick {tick}, sleeping...")
                continue

            iter_ = _get_run_context_iter(run_context)

            # If a new iteration is seen, an attempt at creating a full
            # snapshot event is made. If it can't be created, it is retried
            # until it can.
            # NOTE: there's no timeout for this operation.
            if current_iter != iter_:
                full_snapshot_event = self._full_snapshot_event(iter_)
                if full_snapshot_event is None:
                    logger.debug(
                        f"no full_snapshot_event on new iter {iter_} (current {current_iter}), sleeping at tick {tick}"
                    )
                    continue
                yield full_snapshot_event
                current_iter = iter_
                yield from self._retroactive_update_event()

            self._set_iter_queue(iter_, self._model._job_queue)
            if self._general_interval > 0 and (tick % self._general_interval == 0):
                yield self._partial_snapshot_event(iter_)
            if self._detailed_interval > 0 and (tick % self._detailed_interval == 0):
                yield self._partial_snapshot_event(iter_, read_from_disk=True)
            tick += 1

        yield from self._retroactive_update_event()
        yield self._end_event()

        self._model.teardown_context()

    def _create_snapshot_dict(
        self,
        run_context: ErtRunContext,
        detailed_progress: typing.Tuple[typing.Dict, int],
        iter_: int,
    ) -> typing.Optional[SnapshotDict]:
        """create a snapshot of a run_context and detailed_progress.
        detailed_progress is expected to be a tuple of a realization_progress
        dict and iteration number. iter_ represents the current assimilation
        cycle."""
        self._set_iter_queue(iter_, self._model._job_queue)

        snapshot = SnapshotDict(
            status=ENSEMBLE_STATE_STARTED,
            reals={},
            metadata={"iter": iter_},
        )

        forward_model = self._model.get_forward_model()

        iter_to_progress, progress_iter = detailed_progress
        if progress_iter != iter_:
            logger.debug(
                f"run_context iter ({iter_}) and detailed_progress ({progress_iter} iter differed"
            )

        if iter_ in self._iter_queue and self._iter_queue[iter_] is not None:
            queue_snapshot = self._iter_queue[iter_].snapshot()
        else:
            queue_snapshot = None

        enumerated = 0
        for iens, run_arg in _enumerate_run_context(run_context):
            real_id = str(iens)
            enumerated += 1
            if not _is_iens_active(iens, run_context):
                continue

            status = JobStatusType.JOB_QUEUE_UNKNOWN
            if queue_snapshot is not None and iens in queue_snapshot:
                status = JobStatusType.from_string(queue_snapshot[iens])

            snapshot.reals[real_id] = Realization(
                status=queue_status_to_real_state(status), active=True, steps={}
            )

            step = Step(status="", jobs={})
            snapshot.reals[real_id].steps["0"] = step

            for index in range(0, len(forward_model)):
                ext_job = forward_model.iget_job(index)
                step.jobs[str(index)] = Job(
                    name=ext_job.name(), status=JOB_STATE_START, data={}
                )

            progress = iter_to_progress[iter_].get(iens, None)
            if not progress:
                continue

            jobs = progress[0]
            for idx, fm in enumerate(jobs):
                job = step.jobs[str(idx)]

                job.start_time = fm.start_time
                job.end_time = fm.end_time
                job.name = fm.name
                job.status = _map_job_state(fm.status)
                job.error = fm.error
                job.stdout = fm.std_out_file
                job.stderr = fm.std_err_file
                job.data = {
                    CURRENT_MEMORY_USAGE: fm.current_memory_usage,
                    MAX_MEMORY_USAGE: fm.max_memory_usage,
                }

        if enumerated == 0:
            logger.debug("enumerated 0 items from run_context, it is gone")
            return None

        return snapshot

    def _retroactive_update_event(self):
        """Return generator producing update events for all queues that has run
        thus far."""
        detailed_progress = self._model.getDetailedProgress()
        for iter_ in self._iter_queue:
            partial = self._create_partial_snapshot(None, detailed_progress, iter_)

            if partial is not None:
                self._set_iter_snapshot(iter_, partial._snapshot)

            yield SnapshotUpdateEvent(
                phase_name=self._model.getPhaseName(),
                current_phase=self._model.currentPhase(),
                total_phases=self._model.phaseCount(),
                indeterminate=self._model.isIndeterminate(),
                progress=self._progress(),
                iteration=iter_,
                partial_snapshot=partial,
            )

    def _progress(self) -> float:
        return tracker_progress(self)

    def _set_iter_queue(self, iter_: int, queue: JobQueue) -> None:
        if iter_ < 0:
            return
        if iter_ not in self._iter_queue:
            self._iter_queue[iter_] = None

        if self._iter_queue[iter_] is None and queue is not None:
            # This function might be called when the iteration on the run model
            # has been updated, but the queue has yet to be updated. Hence we check that a
            # queue with the same hash is not found in the map before setting it.
            if hash(queue) not in (hash(q) for q in self._iter_queue.values()):
                self._iter_queue[iter_] = queue

    def _set_iter_snapshot(self, iter_, snapshot: typing.Optional[Snapshot]) -> None:
        if iter_ < 0:
            return
        self._iter_snapshot[iter_] = snapshot

    def _create_partial_snapshot(
        self,
        run_context: ErtRunContext,
        detailed_progress: typing.Tuple[typing.Dict, int],
        iter_: int,
    ) -> typing.Optional[PartialSnapshot]:
        """Create a PartialSnapshot, or None if the sources of data were
        destroyed or had not been created yet. Both run_context and
        detailed_progress needs to be aligned with the stars if job status etc
        is to be produced."""
        queue = self._iter_queue.get(iter_, None)
        if queue is None:
            logger.debug(f"no queue for {iter_}, no partial returned")
            return None

        snapshot = self._iter_snapshot.get(iter_, None)
        if snapshot is None:
            logger.debug(f"no snapshot for {iter_}, no partial returned")
            return None

        partial = PartialSnapshot(snapshot)

        changes = queue.changes_after_transition()
        for iens, change in changes.items():
            change_enum = JobStatusType.from_string(change)
            partial.update_real(
                str(iens),
                Realization(status=queue_status_to_real_state(change_enum)),
            )
        iter_to_progress, progress_iter = detailed_progress
        if not iter_to_progress:
            logger.debug(f"partial: no detailed progress for iter:{iter_}")
            return partial
        if iter_ != progress_iter:
            logger.debug(
                f"partial: iter_to_progress iter ({progress_iter}) differed from run_context ({iter_})"
            )

        for real_id, real in snapshot.get_reals().items():
            if not real.active:
                continue
            progress = iter_to_progress[iter_].get(int(real_id), None)
            if not progress:
                continue

            jobs = progress[0]
            for idx, fm in enumerate(jobs):
                partial.update_job(
                    real_id,
                    "0",
                    str(idx),
                    Job(
                        status=_map_job_state(fm.status),
                        start_time=fm.start_time,
                        end_time=fm.end_time,
                        data={
                            CURRENT_MEMORY_USAGE: fm.current_memory_usage,
                            MAX_MEMORY_USAGE: fm.max_memory_usage,
                        },
                        stdout=fm.std_out_file,
                        stderr=fm.std_err_file,
                        error=fm.error,
                    ),
                )

        return partial

    def _partial_snapshot_event(
        self, iter_, read_from_disk=False
    ) -> SnapshotUpdateEvent:
        """Return a SnapshotUpdateEvent. If read_from_disk is set, this method
        will ultimately read status.json files from disk in order to create an
        event."""
        run_context = self._model.get_run_context()
        detailed_progress = (
            self._model.getDetailedProgress()
            if read_from_disk
            else _THE_EMPTY_DETAILED_PROGRESS
        )
        partial = self._create_partial_snapshot(run_context, detailed_progress, iter_)
        if partial is not None:
            self._set_iter_snapshot(iter_, partial._snapshot)

        return SnapshotUpdateEvent(
            phase_name=self._model.getPhaseName(),
            current_phase=self._model.currentPhase(),
            total_phases=self._model.phaseCount(),
            indeterminate=self._model.isIndeterminate(),
            progress=self._progress(),
            iteration=iter_,
            partial_snapshot=partial,
        )

    def _full_snapshot_event(self, iter_) -> typing.Optional[FullSnapshotEvent]:
        """Return a FullSnapshotEvent if it was possible to create a snapshot.
        Return None if not, indicating that there should be no event."""
        run_context = self._model.get_run_context()
        detailed_progress = self._model.getDetailedProgress()
        if detailed_progress == _THE_EMPTY_DETAILED_PROGRESS:
            return None
        snapshot_dict = self._create_snapshot_dict(
            run_context, detailed_progress, iter_
        )
        if not snapshot_dict:
            return None

        snapshot = Snapshot(snapshot_dict.dict())

        self._set_iter_snapshot(iter_, snapshot)

        return FullSnapshotEvent(
            phase_name=self._model.getPhaseName(),
            current_phase=self._model.currentPhase(),
            total_phases=self._model.phaseCount(),
            indeterminate=self._model.isIndeterminate(),
            progress=self._progress(),
            iteration=iter_,
            snapshot=snapshot,
        )

    def _end_event(self) -> EndEvent:
        return EndEvent(
            failed=self._model.hasRunFailed(), failed_msg=self._model.getFailMessage()
        )

    def is_finished(self) -> bool:
        return self._model.isFinished()

    def request_termination(self) -> None:
        return self._model.killAllSimulations()

    def reset(self):
        self._iter_queue = {}
        self._iter_snapshot = {}


def _get_run_context_iter(run_context: ErtRunContext) -> int:
    """Return the iter from run_context."""
    try:
        return run_context.get_iter()
    except AttributeError:
        return -1


def _is_iens_active(iens: int, run_context: ErtRunContext) -> bool:
    """Return whether or not the iens is active."""
    try:
        return run_context.is_active(iens)
    except AttributeError:
        return False


def _enumerate_run_context(run_context: ErtRunContext) -> typing.Iterable:
    """Return an iterable that's either (iens, run_arg) or empty."""
    try:
        yield from enumerate(run_context)
    except TypeError:
        yield from ()
