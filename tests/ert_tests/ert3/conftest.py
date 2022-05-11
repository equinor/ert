import json
import os
import pathlib
import stat
import sys
import tempfile
import contextlib

import ert.storage

import requests
import pytest
import typing

from ert.data import (
    InMemoryRecordTransmitter,
    SharedDiskRecordTransmitter,
)
from ert.storage import StorageRecordTransmitter

import ert3

from ert_utils import chdir

_EXPERIMENTS_BASE = ert3.workspace._workspace._EXPERIMENTS_BASE

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
    return {
        "polynomial_output": tuple(
            coefficients["a"] * x ** 2 + coefficients["b"] * x + coefficients["c"]
            for x in range(10)
        )
    }
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
def storage_path(workspace, ert_storage):
    yield ert.storage.get_records_url(workspace.name)


@contextlib.contextmanager
def shared_disk_factory_context(**kwargs):
    tmp_path = tempfile.TemporaryDirectory()
    tmp_storage_path = pathlib.Path(tmp_path.name) / ".shared-storage"
    tmp_storage_path.mkdir(parents=True)

    def shared_disk_factory(name: str) -> SharedDiskRecordTransmitter:
        return SharedDiskRecordTransmitter(
            name=name,
            storage_path=tmp_storage_path,
        )

    try:
        yield shared_disk_factory
    finally:
        tmp_path.cleanup()


@contextlib.contextmanager
def in_memory_factory_context(**kwargs):
    def in_memory_factory(name: str) -> InMemoryRecordTransmitter:
        return InMemoryRecordTransmitter(name=name)

    yield in_memory_factory


@contextlib.contextmanager
def ert_storage_factory_context(storage_path):
    def ert_storage_factory(name: str) -> StorageRecordTransmitter:
        return StorageRecordTransmitter(name=name, storage_url=storage_path)

    yield ert_storage_factory


def pytest_generate_tests(metafunc):
    if "record_transmitter_factory_context" in metafunc.fixturenames:
        metafunc.parametrize(
            "record_transmitter_factory_context",
            (
                ert_storage_factory_context,
                in_memory_factory_context,
                shared_disk_factory_context,
            ),
        )


@pytest.fixture()
def workspace_integration(tmpdir):
    from ert_shared.services import Storage

    workspace_dir = pathlib.Path(tmpdir / "polynomial")
    workspace_dir.mkdir()
    with chdir(workspace_dir):

        with Storage.start_server(timeout=120):
            workspace_obj = ert3.workspace.initialize(workspace_dir)
            ert.storage.init(workspace_name=workspace_obj.name)
            yield workspace_obj


@pytest.fixture()
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


@pytest.fixture()
def workspace(tmpdir, ert_storage):
    workspace_dir = pathlib.Path(tmpdir / "polynomial")
    workspace_dir.mkdir()
    with chdir(workspace_dir):
        workspace_obj = ert3.workspace.initialize(workspace_dir)
        ert.storage.init(workspace_name=workspace_obj.name)
        yield workspace_obj


def _create_coeffs_record_file(workspace):
    doe_dir = workspace._path / _EXPERIMENTS_BASE / "doe"
    doe_dir.mkdir(parents=True)
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
    file_path = workspace._path / _EXPERIMENTS_BASE / "doe" / "record.bin"
    file_path.parent.mkdir(parents=True)
    with open(file_path, "wb") as f:
        f.write(b"0x410x420x43")
    return file_path


@pytest.fixture()
def oat_compatible_record_file(workspace_integration):
    sensitivity_dir = (
        workspace_integration._path / _EXPERIMENTS_BASE / "partial_sensitivity"
    )
    sensitivity_dir.mkdir(parents=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(6)]
    with open(sensitivity_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield sensitivity_dir / "coefficients_record.json"


@pytest.fixture()
def oat_incompatible_record_file(workspace_integration):
    sensitivity_dir = (
        workspace_integration._path / _EXPERIMENTS_BASE / "partial_sensitivity"
    )
    sensitivity_dir.mkdir(parents=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(10)]
    with open(sensitivity_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield sensitivity_dir / "coefficients_record.json"


@pytest.fixture()
def base_ensemble_dict():
    yield {
        "size": 10,
        "input": [{"source": "stochastic.coefficients", "name": "coefficients"}],
        "output": [{"name": "polynomial_output"}],
        "forward_model": {"driver": "local", "stage": "evaluate_polynomial"},
    }


@pytest.fixture()
def ensemble(base_ensemble_dict, plugin_registry):
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def stages_config_list():
    yield [
        {
            "name": "evaluate_polynomial",
            "input": [
                {
                    "name": "coefficients",
                    "transformation": {
                        "location": "coefficients.json",
                        "type": "serialization",
                    },
                }
            ],
            "output": [
                {
                    "name": "polynomial_output",
                    "transformation": {
                        "location": "output.json",
                        "type": "serialization",
                    },
                }
            ],
            "script": ["poly --coefficients coefficients.json --output output.json"],
            "transportable_commands": [
                {
                    "name": "poly",
                    "location": "poly.py",
                }
            ],
        }
    ]


@pytest.fixture()
def stages_config(stages_config_list, plugin_registry):
    script_file = pathlib.Path("poly.py")
    script_file.write_text(POLY_SCRIPT)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)

    yield ert3.config.load_stages_config(
        stages_config_list, plugin_registry=plugin_registry
    )

    script_file.unlink()


@pytest.fixture()
def double_stages_config_list():
    yield [
        {
            "name": "evaluate_polynomial",
            "input": [
                {
                    "name": "coefficients",
                    "transformation": {
                        "location": "coefficients.json",
                        "type": "serialization",
                    },
                },
                {
                    "name": "other_coefficients",
                    "transformation": {
                        "location": "other_coefficients.json",
                        "type": "serialization",
                    },
                },
            ],
            "output": [
                {
                    "name": "polynomial_output",
                    "transformation": {
                        "location": "output.json",
                        "type": "serialization",
                    },
                },
                {
                    "name": "other_polynomial_output",
                    "transformation": {
                        "location": "other_output.json",
                        "type": "serialization",
                    },
                },
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


@pytest.fixture()
def double_stages_config(double_stages_config_list, plugin_registry):
    script_file = pathlib.Path("poly.py")
    script_file.write_text(POLY_SCRIPT)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)

    yield ert3.config.load_stages_config(
        double_stages_config_list, plugin_registry=plugin_registry
    )

    script_file.unlink()


@pytest.fixture()
def x_uncertainty_stages_config(plugin_registry):
    config_list = [
        {
            "name": "evaluate_x_uncertainty_polynomial",
            "input": [
                {
                    "name": "coefficients",
                    "transformation": {
                        "location": "coefficients.json",
                        "type": "serialization",
                    },
                },
                {
                    "name": "x_uncertainties",
                    "transformation": {
                        "location": "x_uncertainties.json",
                        "type": "serialization",
                    },
                },
            ],
            "output": [
                {
                    "name": "polynomial_output",
                    "transformation": {
                        "location": "output.json",
                        "type": "serialization",
                    },
                }
            ],
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

    yield ert3.config.load_stages_config(config_list, plugin_registry=plugin_registry)

    script_file.unlink()


@pytest.fixture()
def function_stages_config(plugin_registry):
    config_list = [
        {
            "name": "evaluate_polynomial",
            "input": [
                {
                    "name": "coefficients",
                }
            ],
            "output": [
                {
                    "name": "polynomial_output",
                }
            ],
            "function": "function_steps.functions:polynomial",
        }
    ]
    func_dir = pathlib.Path("function_steps")
    func_dir.mkdir(exist_ok=True)
    (func_dir / "__init__.py").write_text("")
    (func_dir / "functions.py").write_text(POLY_FUNCTION)
    sys.path.append(os.getcwd())

    yield ert3.config.load_stages_config(config_list, plugin_registry=plugin_registry)


@pytest.fixture
def ert_storage(ert_storage_client, monkeypatch):
    # ert_storage_client fixture is defined in ert-storage repo.
    from contextlib import contextmanager
    from ert.storage import _storage
    from httpx import AsyncClient

    @contextmanager
    def _client():
        yield ert_storage_client

    class MockStorageService:
        @staticmethod
        def session():
            return _client()

        @staticmethod
        def async_session():
            return AsyncClient(base_url="http://127.0.0.1")

    ert_storage_client.raise_on_client_error = False
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "ON")
    monkeypatch.setattr(_storage, "Storage", MockStorageService)


@pytest.fixture
def raw_ensrec_to_records():
    def _coerce_raw_ensrec(spec) -> typing.Tuple[ert.data.Record, ...]:
        recs: typing.List[ert.data.Record] = []
        for rec in spec:
            data = rec["data"]
            if isinstance(data, bytes):
                recs.append(ert.data.BlobRecord(data))
            else:
                recs.append(ert.data.NumericalRecord(data=data, index=rec.get("index")))
        return tuple(recs)

    return _coerce_raw_ensrec


@pytest.fixture()
def plugin_registry():
    plugin_registry = ert3.config.ConfigPluginRegistry()
    plugin_registry.register_category(
        category="transformation",
        descriminator="type",
        optional=True,
        base_config=ert3.config.plugins.TransformationConfigBase,
    )
    plugin_manager = ert3.plugins.ErtPluginManager(
        plugins=[ert3.config.plugins.implementations]
    )
    plugin_manager.collect(registry=plugin_registry)
    yield plugin_registry
