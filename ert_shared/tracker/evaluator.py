from ert_shared.tracker.events import GeneralEvent, EndEvent
import time

from cloudevents.http.json_methods import from_json

from ert_shared.tracker.base import BaseTracker
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.tool import recursive_update


class EvaluatorTracker(BaseTracker):
    """The EvaluatorTracker provides tracking of the evaluator."""
    def __init__(self, ee_monitor):
        self._ee_monitor = ee_monitor
        self._snapshot = {}
        # no call to super.__init__ since it's assuming a model, which we don't
        # rely upon.
        self._bootstrap_states()

    def _get_ensemble_size(self):
        return len(self._snapshot["reals"])

    def _count_in_state(self, state):
        if state == "Finished":
            state = "Success"
        count = 0
        for real in self._snapshot["reals"].values():
            for stage in real["stages"].values():
                for step in stage["steps"].values():
                    if state.lower() == step["status"].lower():
                        print("comparing", state.lower(), step["status"].lower())
                        count += 1
        return count

    def track(self):
        start = time.time()
        phase_name = "Some phase..."
        phase = 1
        phase_count = 1
        progress = 0
        for event in self._ee_monitor.track():
            if event["type"] == ids.EVTYPE_EE_SNAPSHOT:
                phase_name = "Got snapshot"
                self._snapshot = event.data
            elif event["type"] == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
                phase_name = "Got update"
                recursive_update(self._snapshot, event.data)
            elif event["type"] == ids.EVTYPE_EE_TERMINATE_REQUEST:
                yield EndEvent(False, None)
                return

            done_count = 0
            for state in self.get_states():
                state.count = self._count_in_state(state.name)
                state.total_count = self._get_ensemble_size()
                if state.name == "Finished":
                    done_count = state.count

                # for queue_state in queue_status:
                #     if queue_state in state.state:
                #         state.count += queue_status[queue_state]

                # if state.name == "Finished":
                #     done_count = state.count


            yield GeneralEvent(
                phase_name,
                phase,
                phase_count,
                progress,
                False,
                self.get_states(),
                time.time() - start,
            )
