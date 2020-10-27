import asyncio
import queue
import threading
from ert_shared.tracker.events import GeneralEvent, EndEvent
import time

from cloudevents.http.json_methods import from_json

from ert_shared.tracker.base import BaseTracker
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.tool import recursive_update
import queue

from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor

"""
monitor provides heart beat so that the updating of UIs can be based off of
those events.

we would like to move out tickevent from tracker, so that 
 - the cli handles updating outside events on its own
 - the gui updates its runtime thing on its own

"""


class EvaluatorTracker(BaseTracker):
    """The EvaluatorTracker provides tracking of the evaluator."""
    def __init__(self, ee_monitor_connection_details, model):
        self._ee_monitor_connection_details = ee_monitor_connection_details
        self._snapshot = {}
        # no call to super.__init__ since it's assuming a model, which we don't
        # rely upon.
        self._bootstrap_states()
        self._q = queue.Queue()

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
                        count += 1
        return count

    async def _fake_tick(self):
        while True:
            await asyncio.sleep(1)
            asyncio.get_event_loop().run_in_executor(None, lambda: self._q.put({"event": "_tick"}))

    async def _model_monitor(self):
        while True:
            await asyncio.sleep(5)
            asyncio.get_event_loop().run_in_executor(None, lambda: self._q.put({"event": "_tick"}))

    def _general_event(self):
        self._update_phase_map()

    def _drain_monitor(self):
        for event in monitor.track():
            if event["type"] == ids.EVTYPE_EE_SNAPSHOT:
                phase_name = "Got snapshot"
                self._snapshot = event.data
                
                yield GeneralEvent(phase_name, phase, phase_count, progress, False, self.get_states(), 
            elif event["type"] == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
                phase_name = "Got update"
                recursive_update(self._snapshot, event.data)
            elif event["type"] == ids.EVTYPE_EE_TERMINATED:
                phase_name = "Done"
                progress = 1

            done_count = 0
            for state in self.get_states():
                state.count = self._count_in_state(state.name)
                state.total_count = self._get_ensemble_size()
                if state.name == "Finished":
                    done_count = state.count

            yield GeneralEvent(
                phase_name,
                phase,
                phase_count,
                progress,
                False,
                self.get_states(),
                time.time() - start,
            )

            if event["type"] == ids.EVTYPE_EE_TERMINATED:
                yield EndEvent(False, None)
                return


            elif event["type"] == "_tick":
                def _tick_event(self):
                    if self._model.stop_time() < self._model.start_time():
                        runtime = time.time() - self._model.start_time()
                    else:
                        runtime = self._model.stop_time() - self._model.start_time()

                    return TickEvent(runtime)




    def track(self):

        start = time.time()
        phase_name = "Some phase..."
        phase = 1
        phase_count = 1
        progress = 0
        while True:
            monitor = create_ee_monitor(self._ee_monitor_connection_details)


                # for queue_state in queue_status:
                #     if queue_state in state.state:
                #         state.count += queue_status[queue_state]

                # if state.name == "Finished":
                #     done_count = state.count




        # phase_name = self._model.getPhaseName()
        # phase = self._model.currentPhase()
        # phase_count = self._model.phaseCount()
        # queue_status = self._model.getQueueStatus()

        # done_count = 0
        # for state in self.get_states():
        #     state.count = 0
        #     state.total_count = self._model.getQueueSize()

        #     for queue_state in queue_status:
        #         if queue_state in state.state:
        #             state.count += queue_status[queue_state]

        #     if state.name == "Finished":
        #         done_count = state.count

        # progress = calculate_progress(
        #     phase,
        #     phase_count,
        #     self._model.isFinished(),
        #     self._model.isQueueRunning(),
        #     self._model.getQueueSize(),
        #     self._phase_states[phase],
        #     done_count,
        # )

        # tick = self._tick_event()
        # return GeneralEvent(
        #     phase_name,
        #     phase,
        #     phase_count,
        #     progress,
        #     self._model.isIndeterminate(),
        #     self.get_states(),
        #     tick.runtime,
        # )
