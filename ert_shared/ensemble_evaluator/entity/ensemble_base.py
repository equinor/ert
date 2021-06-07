import logging

from ert_shared.ensemble_evaluator.client import Client

from ert_shared.ensemble_evaluator.entity import serialization

from cloudevents.http import to_json

from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    Job,
    Realization,
    SnapshotDict,
    Step,
)
import ert_shared.status.entity.state as state

logger = logging.getLogger(__name__)


class _EnsembleStateTracker:
    def __init__(self, state=state.ENSEMBLE_STATE_UNKNOWN):
        self._state: str = state
        self._handles: dict = {}
        self._msg = "Illegal state transition from {} to {}"

        self.set_default_handles()

    def add_handle(self, state, handle):
        self._handles[state] = handle

    def _handle_unknown(self):
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg.format(self._state, state.ENSEMBLE_STATE_UNKNOWN))
        self._state = state.ENSEMBLE_STATE_UNKNOWN

    def _handle_started(self):
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg.format(self._state, state.ENSEMBLE_STATE_STARTED))
        self._state = state.ENSEMBLE_STATE_STARTED

    def _handle_failed(self):
        if self._state not in [
            state.ENSEMBLE_STATE_UNKNOWN,
            state.ENSEMBLE_STATE_STARTED,
        ]:
            logger.warning(self._msg.format(self._state, state.ENSEMBLE_STATE_FAILED))
        self._state = state.ENSEMBLE_STATE_FAILED

    def _handle_stopped(self):
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg.format(self._state, state.ENSEMBLE_STATE_STOPPED))
        self._state = state.ENSEMBLE_STATE_STOPPED

    def _handle_canceled(self):
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(
                self._msg.format(self._state, state.ENSEMBLE_STATE_CANCELLED)
            )
        self._state = state.ENSEMBLE_STATE_CANCELLED

    def set_default_handles(self):
        self.add_handle(state.ENSEMBLE_STATE_UNKNOWN, self._handle_unknown)
        self.add_handle(state.ENSEMBLE_STATE_STARTED, self._handle_started)
        self.add_handle(state.ENSEMBLE_STATE_FAILED, self._handle_failed)
        self.add_handle(state.ENSEMBLE_STATE_STOPPED, self._handle_stopped)
        self.add_handle(state.ENSEMBLE_STATE_CANCELLED, self._handle_canceled)

    def update_state(self, state):
        if state not in self._handles:
            raise KeyError(f"Handle not defined for state {state}")

        # Call the state handle mapped to the new state
        self._handles[state]()

        return self._state


class _Ensemble:
    def __init__(self, reals, metadata):
        self._reals = reals
        self._metadata = metadata
        self._snapshot = self._create_snapshot()
        self._status = self._snapshot.get_status()
        self._status_tracker = _EnsembleStateTracker(self._snapshot.get_status())

    def __repr__(self):
        return f"Ensemble with {len(self._reals)} members"

    def evaluate(self, config, ee_id):
        pass

    def cancel(self):
        pass

    def is_cancellable(self):
        return False

    def get_reals(self):
        return self._reals

    def get_active_reals(self):
        return list(filter(lambda real: real.is_active(), self._reals))

    def get_metadata(self):
        return self._metadata

    @property
    def snapshot(self):
        return self._snapshot

    def update_snapshot(self, events):
        snapshot_mutate_event = PartialSnapshot(self._snapshot)
        for event in events:
            snapshot_mutate_event.from_cloudevent(event)
        self._snapshot.merge_event(snapshot_mutate_event)
        if self._status != self._snapshot.get_status():
            self._status = self._status_tracker.update_state(
                self._snapshot.get_status()
            )
        return snapshot_mutate_event

    def get_status(self):
        return self._status

    async def send_cloudevent(self, url, event, token=None, cert=None, retries=1):
        client = Client(url, token, cert)
        await client._send(
            to_json(event, data_marshaller=serialization.evaluator_marshaller)
        )
        await client.websocket.close()

    def get_successful_realizations(self):
        return self._snapshot.get_successful_realizations()

    def _create_snapshot(self):
        reals = {}
        for real in self.get_active_reals():
            reals[str(real.get_iens())] = Realization(
                active=True,
                status=state.REALIZATION_STATE_WAITING,
            )
            for step in real.get_steps():
                reals[str(real.get_iens())].steps[str(step.get_id())] = Step(
                    status=state.STEP_STATE_UNKNOWN
                )
                for job in step.get_jobs():
                    reals[str(real.get_iens())].steps[str(step.get_id())].jobs[
                        str(job.get_id())
                    ] = Job(
                        status=state.JOB_STATE_START,
                        data={},
                        name=job.get_name(),
                    )
        top = SnapshotDict(
            reals=reals,
            status=state.ENSEMBLE_STATE_UNKNOWN,
            metadata=self.get_metadata(),
        )

        return Snapshot(top.dict())
