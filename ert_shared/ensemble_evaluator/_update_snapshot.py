from typing import List
from cloudevents.http import CloudEvent
from ert.ensemble_evaluator.snapshot import Snapshot, PartialSnapshot

def _calculate_updated_snapshot(snapshot: Snapshot, events: List[CloudEvent]) -> (Snapshot, PartialSnapshot):
#    return ensemble.update_snapshot(events)
    snapshot_mutate_event = PartialSnapshot(snapshot)
    for event in events:
        snapshot_mutate_event.from_cloudevent(event)
    snapshot.merge_event(snapshot_mutate_event)
#    if self.status != self._snapshot.status:
#        self.status = self._status_tracker.update_state(self._snapshot.status)
    return (snapshot, snapshot_mutate_event)
