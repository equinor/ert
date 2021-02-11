import websockets
import pytest
import asyncio
import threading
from unittest.mock import Mock
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FINISHED,
    JOB_STATE_RUNNING,
)
from cloudevents.http import to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    ee_monitor,
)
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.entity.ensemble import (
    create_ensemble_builder,
    create_realization_builder,
    create_stage_builder,
    create_step_builder,
    create_legacy_job_builder,
)
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot


@pytest.fixture
def ee_config(unused_tcp_port):
    return EvaluatorServerConfig(unused_tcp_port)


@pytest.fixture
def evaluator(ee_config):
    ensemble = (
        create_ensemble_builder()
        .add_realization(
            real=create_realization_builder()
            .active(True)
            .set_iens(0)
            .add_stage(
                stage=create_stage_builder()
                .add_step(
                    step=create_step_builder()
                    .set_id(0)
                    .add_job(
                        job=create_legacy_job_builder()
                        .set_id(0)
                        .set_name("cat")
                        .set_ext_job(Mock())
                    )
                    .add_job(
                        job=create_legacy_job_builder()
                        .set_id(1)
                        .set_name("cat2")
                        .set_ext_job(Mock())
                    )
                    .set_dummy_io()
                )
                .set_id(0)
                .set_status("Unknown")
            )
        )
        .set_ensemble_size(2)
        .build()
    )
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    yield ee
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
                    msg = await q.get()
                    if msg == "stop":
                        return
                    await websocket.send(msg)

        loop.run_until_complete(send_loop(self.q))

    def send(self, msg):
        self.loop.call_soon_threadsafe(self.q.put_nowait, msg)

    def stop(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, "stop")
        self.thread.join()
        self.loop.close()


def send_dispatch_event(client, event_type, source, event_id, data):
    event1 = CloudEvent({"type": event_type, "source": source, "id": event_id}, data)
    client.send(to_json(event1))


def test_dispatchers_can_connect_and_monitor_can_shut_down_evaluator(evaluator):
    with evaluator.run() as monitor:
        events = monitor.track()

        host = evaluator._config.host
        port = evaluator._config.port

        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
        # two dispatchers connect
        with Client(host, port, "/dispatch") as dispatch1, Client(
            host, port, "/dispatch"
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
            assert snapshot.get_job("0", "0", "0", "0").status == JOB_STATE_RUNNING

            # second dispatcher informs that job 0 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                "/ert/ee/0/real/1/stage/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("1", "0", "0", "0").status == JOB_STATE_RUNNING

            # second dispatcher informs that job 0 is done
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_SUCCESS,
                "/ert/ee/0/real/1/stage/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("1", "0", "0", "0").status == JOB_STATE_FINISHED

            # a second monitor connects
            with ee_monitor.create(host, port) as monitor2:
                events2 = monitor2.track()
                snapshot = Snapshot(next(events2).data)
                assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
                assert snapshot.get_job("0", "0", "0", "0").status == JOB_STATE_RUNNING
                assert snapshot.get_job("1", "0", "0", "0").status == JOB_STATE_FINISHED

                # one monitor requests that server exit
                monitor.signal_cancel()

                # both monitors should get a terminated event
                terminated = next(events)
                terminated2 = next(events2)
                assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
                assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

                for e in [events, events2]:
                    for _ in e:
                        assert False, "got unexpected event from monitor"


def test_monitor_stop(evaluator):
    with evaluator.run() as monitor:
        for event in monitor.track():
            snapshot = Snapshot(event.data)
            break
    assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
