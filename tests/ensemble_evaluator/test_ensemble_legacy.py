import threading
import websockets
import pytest
import asyncio
import logging
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from pathlib import Path
from ert_shared.ensemble_evaluator.config import CONFIG_FILE, load_config
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.ws_util import wait as wait_for_ws
from cloudevents.http.event import CloudEvent
from cloudevents.http import to_json


def test_run_legacy_ensemble(tmpdir, unused_tcp_port, make_ensemble_builder):
    num_reals = 2
    conf_file = Path(tmpdir / CONFIG_FILE)
    with tmpdir.as_cwd():
        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = load_config(conf_file)
        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")
        monitor = evaluator.run()
        for e in monitor.track():
            if (
                e["type"]
                in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                )
                and e.data.get("status") == "Stopped"
            ):
                monitor.signal_done()
        assert evaluator.get_successful_realizations() == num_reals


@pytest.mark.asyncio
async def test_run_and_cancel_legacy_ensemble(
    tmpdir, unused_tcp_port, make_ensemble_builder
):
    num_reals = 10
    conf_file = Path(tmpdir / CONFIG_FILE)

    with tmpdir.as_cwd():
        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = load_config(conf_file)

        evaluator = EnsembleEvaluator(ensemble, config, ee_id="1")

        thread = threading.Thread(
            name="test_eval",
            target=evaluator.run_and_get_successful_realizations,
            args=(),
        )
        thread.start()

        # Wait for evaluator to start
        await wait_for_ws(config["url"], 10)

        # Send termination request to the evaluator
        async with websockets.connect(config["client_url"]) as websocket:
            out_cloudevent = CloudEvent(
                {
                    "type": identifiers.EVTYPE_EE_USER_CANCEL,
                    "source": "/ert/test/0",
                    "id": "ID",
                }
            )
            await websocket.send(to_json(out_cloudevent))

        thread.join()
        assert evaluator._snapshot.get_status() == "Cancelled"
