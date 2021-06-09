import os
import stat

import json
import pytest
import sys

import pathlib
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


@pytest.fixture()
def workspace(tmpdir, ert_storage):
    workspace = tmpdir / "polynomial"
    workspace.mkdir()
    workspace.chdir()
    ert3.workspace.initialize(workspace)
    yield workspace


@pytest.fixture()
def designed_coeffs_record_file(workspace):
    doe_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "doe"
    doe_dir.ensure(dir=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(10)]
    with open(doe_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield doe_dir / "coefficients_record.json"


@pytest.fixture()
def base_ensemble_dict():
    yield {
        "size": 10,
        "input": [{"source": "stochastic.coefficients", "record": "coefficients"}],
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
    func_dir.mkdir()
    (func_dir / "__init__.py").write_text("")
    (func_dir / "functions.py").write_text(POLY_FUNCTION)
    sys.path.append(os.getcwd())

    yield ert3.config.load_stages_config(config_list)


@pytest.fixture
def ert_storage(ert_storage_client, monkeypatch):
    from ert3.storage import _storage

    ert_storage_client.raise_on_client_error = False

    # Fix baseurl prefix
    monkeypatch.setattr(_storage, "_STORAGE_URL", "")

    # Fix requests library
    for func in "get", "post", "put", "delete":
        monkeypatch.setattr(_storage.requests, func, getattr(ert_storage_client, func))
