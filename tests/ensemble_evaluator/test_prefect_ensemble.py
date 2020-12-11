import os
from pathlib import Path
import yaml
import websockets
import threading
import pytest
from cloudevents.http.event import CloudEvent
from cloudevents.http import to_json, from_json

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.config import CONFIG_FILE, load_config
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.ws_util import wait as wait_for_ws
from ert_shared.ensemble_evaluator.entity.prefect_ensemble import PrefectEnsemble
from tests.utils import SOURCE_DIR, tmp


def parse_config(path):
    conf_path = os.path.abspath(path)
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


@pytest.mark.asyncio
async def test_run_prefect_ensemble(tmpdir, unused_tcp_port):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/flow_test_case"), False):
        conf_file = Path(CONFIG_FILE)
        config = parse_config("config.yml")
        config.update({"config_path": Path.absolute(Path("."))})
        config.update({"realizations": 2})
        config.update({"executor": "local"})

        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        service_config = load_config(conf_file)
        config.update(service_config)
        ensemble = PrefectEnsemble(config)

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
            async for msg in websocket:
                client_event = from_json(msg)
                if client_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT_UPDATE:
                    if client_event.data.get("status") == "Stopped":
                        out_cloudevent = CloudEvent(
                            {
                                "type": identifiers.EVTYPE_EE_USER_DONE,
                                "source": "/ert/test/0",
                                "id": "ID",
                            }
                        )
                        await websocket.send(to_json(out_cloudevent))
        thread.join()
        assert evaluator._snapshot.get_status() == "Stopped"

        successful_realizations = evaluator._snapshot.get_successful_realizations()

        assert successful_realizations == config["realizations"]
