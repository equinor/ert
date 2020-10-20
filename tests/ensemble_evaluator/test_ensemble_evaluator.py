from cloudevents.http import to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    ee_entity,
    ee_monitor,
)
import websockets
import pytest
import asyncio
import threading
import json


@pytest.fixture
def evaluator(unused_tcp_port):
    ee = EnsembleEvaluator(port=unused_tcp_port)
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
    event1 = CloudEvent({
        "type": event_type, "source": source, "id": event_id
    }, data)
    client.send(to_json(event1))


def test_dispatchers_can_connect_and_monitor_can_shut_down_evaluator(evaluator):
    monitor = evaluator.run()
    events = monitor.track()

    # first snapshot before any event occurs
    snapshot = next(events)
    assert snapshot.to_dict()["status"] == "unknown"

    # two dispatchers connect
    with Client(evaluator._host, evaluator._port, "/dispatch") as dispatch1, Client(evaluator._host, evaluator._port, "/dispatch") as dispatch2:

        # first dispatcher informs that job 1 is running
        send_dispatch_event(dispatch1, ee_entity._FM_JOB_RUNNING,
        "/ert/ee/0/real/0/stage/0/step/0/job/1", "event1", {"current_memory_usage": 1000})
        connect1 = next(events).to_dict()
        assert connect1["status"] == "running"
        assert connect1["realizations"]["0"]["status"] == "running"
        return
        # second dispatcher informs that job 1 is running
        send_dispatch_event(dispatch2, realization=1, job=1, status="running")
        connect2 = next(events).to_dict()
        assert connect2["status"] == "running"
        assert connect2["realizations"]["1"]["forward_models"]["1"]["status"] == "running"

        # second dispatcher informs that job 1 is done
        send_dispatch_event(dispatch2, realization=1, job=1, status="done")
        connect2 = next(events).to_dict()
        assert connect2["status"] == "running"
        assert connect2["realizations"]["1"]["forward_models"]["1"]["status"] == "done"

        # a second monitor connects
        monitor2 = ee_monitor.create(evaluator._host, evaluator._port)
        events2 = monitor2.track()
        snapshot2 = next(events2).to_dict()
        # second monitor should get the updated snapshot
        assert snapshot2["status"] == "running"
        assert snapshot2["realizations"]["0"]["forward_models"]["1"]["status"] == "running"
        assert snapshot2["realizations"]["1"]["forward_models"]["1"]["status"] == "done"

        # one monitor requests that server exit
        monitor.exit_server()

        # both monitors should get a terminated event
        terminated = next(events)
        terminated2 = next(events2)
        assert terminated.to_dict()["status"] == "terminated"
        assert terminated2.to_dict()["status"] == "terminated"


def test_monitor_stop(evaluator):
    monitor = evaluator.run()
    events = monitor.track()
    snapshot = next(events)
    assert snapshot.to_dict()["status"] == "unknown"
