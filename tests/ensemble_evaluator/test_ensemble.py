import os
import ruamel.yaml as yaml
import websockets
import asyncio
from ert_shared.ensemble_evaluator.entity.prefect_ensamble import PrefectEnsemble
from ert_shared.ensemble_evaluator.config import load_config
import pytest


def parse_config(path):
    conf_path = os.path.abspath(path)
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


# def test_ensemble(tmpdir):
#     service_config = load_config()
#     config = parse_config("../../test-data/local/flow_test_case/config.yml")
#
#     config.update(service_config)
#     ens = PrefectEnsemble(config)
#     with tmpdir.as_cwd():
#         state = ens.evaluate("localhost", 8765)
#         assert state.is_successful()

#
# def test_start_server():
#     async def hello(websocket, path):
#         async for msg in websocket:
#             print(f"< {msg}")
#     start_server = websockets.serve(hello, "localhost", 8765)
#
#     asyncio.get_event_loop().run_until_complete(start_server)
#     asyncio.get_event_loop().run_forever()
