import json
import os
import pathlib
import stat
import sys

import ert.storage
from ert_shared.storage.server_monitor import (
    ServerMonitor,
)

import requests
import pytest
import typing

import ert3

POLY_SCRIPT = """#!/usr/bin/env python3
import json
import sys


def _poly():
    with open(sys.argv[2], "r") as f:
        coefficients = json.load(f)
    a, b, c = coefficients["a"], coefficients["b"], coefficients["c"]
    result = tuple(a * x ** 2 + b * x + c for x in range(10))
    with open(sys.argv[4], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    _poly()

"""

POLY_FUNCTION = """
def polynomial(coefficients):
    return tuple(
        coefficients["a"] * x ** 2 + coefficients["b"] * x + coefficients["c"]
        for x in range(10)
    )
"""

POLY_SCRIPT_X_UNCERTAINTIES = """#!/usr/bin/env python3
import json
import sys


def _poly():
    with open(sys.argv[2], "r") as f:
        coefficients = json.load(f)
    a, b, c = coefficients["a"], coefficients["b"], coefficients["c"]
    with open(sys.argv[4], "r") as f:
        x_uncertainties = json.load(f)
    xs = map(sum, zip(range(10), x_uncertainties))
    result = tuple(a * x ** 2 + b * x + c for x in xs)
    with open(sys.argv[6], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    _poly()

"""


@pytest.fixture()
def workspace_integration(tmpdir):
    workspace = tmpdir / "polynomial"
    workspace.mkdir()
    workspace.chdir()
    server = ServerMonitor()
    server.start()

    resp = requests.get(f"{server.fetch_url()}/healthcheck", auth=server.fetch_auth())
    assert "ALL OK!" in resp.json()

    ert3.workspace.initialize(workspace)
    yield workspace
    server.shutdown()
    ert.storage.StorageInfo._url = None
    ert.storage.StorageInfo._token = None


@pytest.fixture()
def workspace(tmpdir, ert_storage):
    workspace = tmpdir / "polynomial"
    workspace.mkdir()
    workspace.chdir()
    ert3.workspace.initialize(workspace)
    yield workspace
    ert.storage.StorageInfo._url = None
    ert.storage.StorageInfo._token = None


def _create_coeffs_record_file(workspace):
    doe_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "doe"
    doe_dir.ensure(dir=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(10)]
    with open(doe_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    return doe_dir / "coefficients_record.json"


@pytest.fixture()
def designed_coeffs_record_file(workspace):
    yield _create_coeffs_record_file(workspace)


@pytest.fixture()
def designed_coeffs_record_file_integration(workspace_integration):
    yield _create_coeffs_record_file(workspace_integration)


@pytest.fixture()
def designed_blob_record_file(workspace):
    file_path = workspace / ert3.workspace.EXPERIMENTS_BASE / "doe" / "record.bin"
    file_path.dirpath().ensure(dir=True)
    with open(file_path, "wb") as f:
        f.write(b"0x410x420x43")
    return file_path


@pytest.fixture()
def oat_compatible_record_file(workspace_integration):
    sensitivity_dir = (
        workspace_integration / ert3.workspace.EXPERIMENTS_BASE / "partial_sensitivity"
    )
    sensitivity_dir.ensure(dir=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(6)]
    with open(sensitivity_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield sensitivity_dir / "coefficients_record.json"


@pytest.fixture()
def oat_incompatible_record_file(workspace_integration):
    sensitivity_dir = (
        workspace_integration / ert3.workspace.EXPERIMENTS_BASE / "partial_sensitivity"
    )
    sensitivity_dir.ensure(dir=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(10)]
    with open(sensitivity_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield sensitivity_dir / "coefficients_record.json"


@pytest.fixture()
def base_ensemble_dict():
    yield {
        "size": 10,
        "input": [{"source": "stochastic.coefficients", "record": "coefficients"}],
        "output": [{"record": "polynomial_output"}],
        "forward_model": {"driver": "local", "stage": "evaluate_polynomial"},
    }


@pytest.fixture()
def ensemble(base_ensemble_dict):
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def stages_config():
    config_list = [
        {
            "name": "evaluate_polynomial",
            "input": [{"record": "coefficients", "location": "coefficients.json"}],
            "output": [{"record": "polynomial_output", "location": "output.json"}],
            "script": ["poly --coefficients coefficients.json --output output.json"],
            "transportable_commands": [
                {
                    "name": "poly",
                    "location": "poly.py",
                }
            ],
        }
    ]
    script_file = pathlib.Path("poly.py")
    script_file.write_text(POLY_SCRIPT)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)

    yield ert3.config.load_stages_config(config_list)


@pytest.fixture()
def double_stages_config():
    config_list = [
        {
            "name": "evaluate_polynomial",
            "input": [
                {"record": "coefficients", "location": "coefficients.json"},
                {"record": "other_coefficients", "location": "other_coefficients.json"},
            ],
            "output": [
                {"record": "polynomial_output", "location": "output.json"},
                {"record": "other_polynomial_output", "location": "other_output.json"},
            ],
            "script": [
                "poly --coefficients coefficients.json --output output.json",
                (
                    "poly --coefficients other_coefficients.json "
                    "--output other_output.json"
                ),
            ],
            "transportable_commands": [
                {
                    "name": "poly",
                    "location": "poly.py",
                }
            ],
        }
    ]
    script_file = pathlib.Path("poly.py")
    script_file.write_text(POLY_SCRIPT)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)

    yield ert3.config.load_stages_config(config_list)


@pytest.fixture()
def x_uncertainty_stages_config():
    config_list = [
        {
            "name": "evaluate_x_uncertainty_polynomial",
            "input": [
                {"record": "coefficients", "location": "coefficients.json"},
                {"record": "x_uncertainties", "location": "x_uncertainties.json"},
            ],
            "output": [{"record": "polynomial_output", "location": "output.json"}],
            "script": [
                "poly --coefficients coefficients.json \
                    --x_uncertainties x_uncertainties.json --output output.json"
            ],
            "transportable_commands": [
                {
                    "name": "poly",
                    "location": "poly.py",
                }
            ],
        }
    ]
    script_file = pathlib.Path("poly.py")
    script_file.write_text(POLY_SCRIPT_X_UNCERTAINTIES)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)

    yield ert3.config.load_stages_config(config_list)


@pytest.fixture()
def function_stages_config():
    config_list = [
        {
            "name": "evaluate_polynomial",
            "input": [{"record": "coefficients", "location": "coeffs"}],
            "output": [{"record": "polynomial_output", "location": "output"}],
            "function": "function_steps.functions:polynomial",
        }
    ]
    func_dir = pathlib.Path("function_steps")
    func_dir.mkdir(exist_ok=True)
    (func_dir / "__init__.py").write_text("")
    (func_dir / "functions.py").write_text(POLY_FUNCTION)
    sys.path.append(os.getcwd())

    yield ert3.config.load_stages_config(config_list)


@pytest.fixture
def ert_storage(ert_storage_client, monkeypatch):
    from ert.storage import _storage

    ert_storage_client.raise_on_client_error = False
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "ON")
    # Fix requests library
    for func in "get", "post", "put", "delete":
        monkeypatch.setattr(_storage.requests, func, getattr(ert_storage_client, func))

    monkeypatch.setattr(
        _storage,
        "get_info",
        lambda: {"baseurl": "http://127.0.0.1:51820", "auth": ("", "")},
    )


@pytest.fixture
def raw_ensrec_to_records():
    def _coerce_raw_ensrec(spec) -> typing.Tuple[ert.data.Record]:
        recs = []
        for rec in spec:
            data = rec["data"]
            if isinstance(data, bytes):
                recs.append(ert.data.BlobRecord.parse_obj(rec))
            else:
                recs.append(ert.data.NumericalRecord.parse_obj(rec))
        return tuple(recs)

    return _coerce_raw_ensrec
