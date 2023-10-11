import time

from _ert_job_runner.client import Client
from ert.ensemble_evaluator import Snapshot, identifiers
from ert.ensemble_evaluator.monitor import Monitor
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_UNKNOWN,
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_RUNNING,
)

from .ensemble_evaluator_utils import send_dispatch_event


def test_new_monitor_can_pick_up_where_we_left_off(evaluator):
    evaluator._start_running()
    token = evaluator._config.token
    cert = evaluator._config.cert
    url = evaluator._config.url

    config_info = evaluator._config.get_connection_info()
    with Monitor(config_info) as monitor:
        events = monitor.track()
        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        with Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch1, Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch2:
            # first dispatch endpoint client informs that job 0 is running
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            # second dispatch endpoint client informs that job 1 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/1",
                "event1",
                {"current_memory_usage": 1000},
            )

        evt = next(events)
        snapshot = Snapshot(evt.data)
        assert snapshot.get_job("0", "0").status == JOB_STATE_RUNNING
        assert snapshot.get_job("1", "0").status == JOB_STATE_RUNNING
        assert snapshot.get_job("1", "1").status == JOB_STATE_RUNNING
        # take down first monitor by leaving context

    with Client(
        url + "/dispatch",
        cert=cert,
        token=token,
        max_retries=1,
        timeout_multiplier=1,
    ) as dispatch2:
        # second dispatch endpoint client informs that job 0 is done
        send_dispatch_event(
            dispatch2,
            identifiers.EVTYPE_FM_JOB_SUCCESS,
            f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/0",
            "event1",
            {"current_memory_usage": 1000},
        )

        # second dispatch endpoint client informs that job 1 is failed
        send_dispatch_event(
            dispatch2,
            identifiers.EVTYPE_FM_JOB_FAILURE,
            f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/1",
            "event_job_1_fail",
            {identifiers.ERROR_MSG: "error"},
        )

    # we have to wait for the batching dispatcher to process the events, and for the
    # internal ensemble state to get updated before connecting and getting a full
    # ensemble snapshot
    time.sleep(2)
    # reconnect new monitor
    with Monitor(config_info) as new_monitor:
        new_events = new_monitor.track()
        full_snapshot_event = next(new_events)

        assert full_snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
        snapshot = Snapshot(full_snapshot_event.data)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        assert snapshot.get_job("0", "0").status == JOB_STATE_RUNNING
        assert snapshot.get_job("1", "0").status == JOB_STATE_FINISHED
        assert snapshot.get_job("1", "1").status == JOB_STATE_FAILURE


def test_dispatch_endpoint_clients_can_connect_and_monitor_can_shut_down_evaluator(
    evaluator,
):
    evaluator._start_running()
    conn_info = evaluator._config.get_connection_info()
    with Monitor(conn_info) as monitor:
        events = monitor.track()
        token = evaluator._config.token
        cert = evaluator._config.cert

        url = evaluator._config.url
        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        with Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch1, Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch2:
            # first dispatch endpoint client informs that job 0 is running
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is done
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_SUCCESS,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 1 is failed
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_FAILURE,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0/job/1",
                "event_job_1_fail",
                {identifiers.ERROR_MSG: "error"},
            )
            evt = next(events)
            snapshot = Snapshot(evt.data)
            assert snapshot.get_job("1", "0").status == JOB_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == JOB_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == JOB_STATE_FAILURE

        # a second monitor connects
        with Monitor(evaluator._config.get_connection_info()) as monitor2:
            events2 = monitor2.track()
            full_snapshot_event = next(events2)
            assert full_snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
            snapshot = Snapshot(full_snapshot_event.data)
            assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
            assert snapshot.get_job("1", "0").status == JOB_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == JOB_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == JOB_STATE_FAILURE

            # one monitor requests that server exit
            monitor.signal_cancel()

            # both monitors should get a terminated event
            terminated = next(events)
            terminated2 = next(events2)
            assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
            assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

            for e in [events, events2]:
                for undexpected_event in e:
                    raise AssertionError(
                        f"got unexpected event {undexpected_event} from monitor"
                    )


def test_ensure_multi_level_events_in_order(evaluator):
    evaluator._start_running()
    config_info = evaluator._config.get_connection_info()
    with Monitor(config_info) as monitor:
        events = monitor.track()

        token = evaluator._config.token
        cert = evaluator._config.cert
        url = evaluator._config.url

        snapshot_event = next(events)
        assert snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
        with Client(url + "/dispatch", cert=cert, token=token) as dispatch1:
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ensemble/{evaluator.ensemble.id_}",
                "event0",
                {},
            )
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FM_STEP_SUCCESS,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0",
                "event1",
                {},
            )
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FM_STEP_SUCCESS,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0",
                "event2",
                {},
            )
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_ENSEMBLE_STOPPED,
                f"/ert/ensemble/{evaluator.ensemble.id_}",
                "event3",
                {},
            )
        monitor.signal_done()
        events = list(events)

        # Without making too many assumptions about what events to expect, it
        # should be reasonable to expect that if an event contains information
        # about realizations, the state of the ensemble up until that point
        # should be not final (i.e. not cancelled, stopped, failed).
        ensemble_state = snapshot_event.data.get("status")
        for event in events:
            if event.data:
                if "reals" in event.data:
                    assert ensemble_state == ENSEMBLE_STATE_STARTED
                ensemble_state = event.data.get("status", ensemble_state)


def test_dying_batcher(evaluator):
    def exploding_handler(events):
        raise ValueError("Boom!")

    evaluator._dispatcher.set_event_handler({"EXPLODING"}, exploding_handler)

    evaluator._start_running()
    config_info = evaluator._config.get_connection_info()

    with Monitor(config_info) as monitor:
        token = evaluator._config.token
        cert = evaluator._config.cert
        url = evaluator._config.url

        with Client(url + "/dispatch", cert=cert, token=token) as dispatch:
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ensemble/{evaluator.ensemble.id_}",
                "event0",
                {},
            )
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0/job/0",
                "event2",
                {},
            )
            send_dispatch_event(
                dispatch,
                "EXPLODING",
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/1/step/0",
                "event3",
                {},
            )
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_FM_STEP_SUCCESS,
                f"/ert/ensemble/{evaluator.ensemble.id_}/real/0/step/0/job/0",
                "event4",
                {},
            )

        # drain the monitor
        list(monitor.track())

        assert evaluator.ensemble.snapshot.status == ENSEMBLE_STATE_FAILED
