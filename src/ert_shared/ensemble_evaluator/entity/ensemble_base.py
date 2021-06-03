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
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_START,
    REALIZATION_STATE_WAITING,
    STEP_STATE_UNKNOWN,
)


class _Ensemble:
    def __init__(self, reals, metadata):
        self._reals = reals
        self._metadata = metadata
        self._snapshot = self._create_snapshot()

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
        return snapshot_mutate_event

    def get_status(self):
        return self._snapshot.get_status()

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
                status=REALIZATION_STATE_WAITING,
            )
            for step in real.get_steps():
                reals[str(real.get_iens())].steps[str(step.get_id())] = Step(
                    status=STEP_STATE_UNKNOWN
                )
                for job in step.get_jobs():
                    reals[str(real.get_iens())].steps[str(step.get_id())].jobs[
                        str(job.get_id())
                    ] = Job(
                        status=JOB_STATE_START,
                        data={},
                        name=job.get_name(),
                    )
        top = SnapshotDict(
            reals=reals,
            status=ENSEMBLE_STATE_STARTED,
            metadata=self.get_metadata(),
        )

        return Snapshot(top.dict())
