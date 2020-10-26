from cloudevents.http import to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    ee_monitor,
)
from ert_shared.ensemble_evaluator.entity.ensemble import _Ensemble
from ert_shared.ensemble_evaluator.entity.snapshot import SnapshotBuilder
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot
import websockets
import pytest
import asyncio
import threading
import json


class DummyEnsemble:

    def __init__(self, snapshot):
        self._snapshot = snapshot

    def forward_model_description(self):
        return self._snapshot

    def evaluate(self, host, port):
        pass

@pytest.fixture
def evaluator(unused_tcp_port):
    snapshot = (
        SnapshotBuilder()
        .add_stage(stage_id="0", status="unknown")
        .add_step(stage_id="0", step_id="0", status="unknown")
        .add_job(stage_id="0", step_id="0", job_id="0", data={}, status="unknown")
        .build(["0", "1"], status="unknown")
    )
    ensemble = DummyEnsemble(snapshot=snapshot)
    ee = EnsembleEvaluator(ensemble=ensemble, port=unused_tcp_port)
    yield ee
    print("fixture exit")
    ee.stop()


class Client:
    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

    def __init__(self, host, port, path):
        self.host = host
        self.port = port
        self.path = path
        self.loop = asyncio.new_event_loop()
        self.q = asyncio.Queue(loop=self.loop)
        self.thread = threading.Thread(
            name="test_websocket_client", target=self._run, args=(self.loop,)
        )

    def _run(self, loop):
        asyncio.set_event_loop(loop)
        uri = f"ws://{self.host}:{self.port}{self.path}"

        async def send_loop(q):
            async with websockets.connect(uri) as websocket:
                while True:
                    print("waiting for q")
                    msg = await q.get()
                    if msg == "stop":
                        return
                    print(f"sending: {msg}")
                    await websocket.send(msg)

        loop.run_until_complete(send_loop(self.q))

    def send(self, msg):
        self.loop.call_soon_threadsafe(self.q.put_nowait, msg)

    def stop(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, "stop")
        self.thread.join()


def send_dispatch_event(client, event_type, source, event_id, data):
    event1 = CloudEvent({"type": event_type, "source": source, "id": event_id}, data)
    client.send(to_json(event1))


def test_dispatchers_can_connect_and_monitor_can_shut_down_evaluator(evaluator):
    monitor = evaluator.run()
    events = monitor.track()

    # first snapshot before any event occurs
    snapshot_event = next(events)
    snapshot = Snapshot(snapshot_event.data)
    assert snapshot.get_status() == "unknown"

    # two dispatchers connect
    with Client(evaluator._host, evaluator._port, "/dispatch") as dispatch1, Client(
        evaluator._host, evaluator._port, "/dispatch"
    ) as dispatch2:

        # first dispatcher informs that job 0 is running
        send_dispatch_event(
            dispatch1,
            identifiers.EVTYPE_FM_JOB_RUNNING,
            "/ert/ee/0/real/0/stage/0/step/0/job/0",
            "event1",
            {"current_memory_usage": 1000},
        )
        snapshot = Snapshot(next(events).data)
        assert snapshot.get_job("0", "0", "0", "0")["status"] == "running"

        # second dispatcher informs that job 0 is running
        send_dispatch_event(
            dispatch2,
            identifiers.EVTYPE_FM_JOB_RUNNING,
            "/ert/ee/0/real/1/stage/0/step/0/job/0",
            "event1",
            {"current_memory_usage": 1000},
        )
        snapshot = Snapshot(next(events).data)
        assert snapshot.get_job("1", "0", "0", "0")["status"] == "running"

        # second dispatcher informs that job 0 is done
        send_dispatch_event(
            dispatch2,
            identifiers.EVTYPE_FM_JOB_SUCCESS,
            "/ert/ee/0/real/1/stage/0/step/0/job/0",
            "event1",
            {"current_memory_usage": 1000},
        )
        snapshot = Snapshot(next(events).data)
        assert snapshot.get_job("1", "0", "0", "0")["status"] == "success"

        # a second monitor connects
        monitor2 = ee_monitor.create(evaluator._host, evaluator._port)
        events2 = monitor2.track()
        snapshot = Snapshot(next(events2).data)
        assert snapshot.get_status() == "unknown"
        assert snapshot.get_job("0", "0", "0", "0")["status"] == "running"
        assert snapshot.get_job("1", "0", "0", "0")["status"] == "success"

        # one monitor requests that server exit
        monitor.exit_server()

        # both monitors should get a terminated event
        terminated = next(events)
        terminated2 = next(events2)
        assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
        assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

        for e in [events, events2]:
            for _ in e:
                assert False, "got unexpected event from monitor"

    # Make sure evaluator exits properly
    evaluator.stop()


def test_monitor_stop(evaluator):
    monitor = evaluator.run()
    events = monitor.track()
    snapshot = Snapshot(next(events).data)
    assert snapshot.get_status() == "unknown"
