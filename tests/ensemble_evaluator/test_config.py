from ert_shared.ensemble_evaluator.config import (
    load_config,
    CLIENT_URI,
    DISPATCH_URI,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
import pytest
import json


@pytest.mark.parametrize(
    "host, port", [("test_host", "42"), (None, "42"), ("test_host", None), (None, None)]
)
def test_load_config(tmpdir, host, port):
    config_dict = {}
    expected_host = host if host else DEFAULT_HOST
    expected_port = port if port else DEFAULT_PORT
    expected_config = {
        "host": expected_host,
        "port": expected_port,
        "url": f"ws://{expected_host}:{expected_port}",
        "client_url": f"ws://{expected_host}:{expected_port}/{CLIENT_URI}",
        "dispatch_url": f"ws://{expected_host}:{expected_port}/{DISPATCH_URI}",
    }

    if host is not None:
        config_dict["host"] = host
    if port is not None:
        config_dict["port"] = port

    with tmpdir.as_cwd():
        with open("ee_config", "w") as f:
            json.dump(config_dict, f)

        res = load_config("ee_config")
        assert res == expected_config
