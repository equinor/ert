"""Unit tests for the new experiment_server endpoints:
- POST /check_runpath
- POST /delete_runpaths
- POST /start_experiment?rerun_from_run_id={run_id}
- GET  /has_failed_realizations/{run_id}
"""

from __future__ import annotations

import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ert.config import ErtConfig
from ert.dark_storage.app import app
from ert.dark_storage.endpoints.experiment_server import (
    ExperimentRunnerState,
    _runs,
)
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models.model_factory import build_run_model_config
from everest.strings import EverEndpoints

_TOKEN = "test-token"
_AUTH = ("__token__", _TOKEN)

check_runpath_url = f"/experiment_server/{EverEndpoints.check_runpath}"
_DELETE_RUNPATHS_URL = f"/experiment_server/{EverEndpoints.delete_runpaths}"


@pytest.fixture
def experiment_server_client(monkeypatch):
    """TestClient with ERT_STORAGE_TOKEN set."""
    original = dict(_runs)
    monkeypatch.setenv("ERT_STORAGE_TOKEN", _TOKEN)
    with TestClient(app) as client:
        yield client
    _runs.clear()
    _runs.update(original)


@pytest.fixture
def poly_run_model_config(copy_poly_case):
    """Returns a serialized EnsembleExperiment RunModelConfig for poly_example."""
    config = ErtConfig.from_file("poly.ert")
    args = Namespace(
        mode=ENSEMBLE_EXPERIMENT_MODE,
        realizations="0,1",
        experiment_name="preflight_test",
        current_ensemble="default",
    )
    return build_run_model_config(config, args).model_dump(mode="json")


@pytest.fixture
def copy_poly_case(tmp_path, source_root, monkeypatch):
    poly_dir = tmp_path / "poly_example"
    shutil.copytree(
        source_root / "test-data" / "ert" / "poly_example",
        poly_dir,
        ignore=shutil.ignore_patterns("*ipynb", "poly_out", "storage", "logs"),
    )
    monkeypatch.chdir(poly_dir)
    return poly_dir


def test_that_check_runpath_returns_false_when_no_runpath_exists(
    experiment_server_client, poly_run_model_config
):
    response = experiment_server_client.post(
        check_runpath_url, json=poly_run_model_config, auth=_AUTH
    )
    assert response.status_code == 200
    data = response.json()
    assert data["runpath_exists"] is False
    assert data["num_existing"] == 0
    assert data["num_active"] == 2


def test_that_check_runpath_returns_true_when_runpath_exists(
    experiment_server_client, poly_run_model_config, tmp_path
):
    # Create the runpath directories that the config expects

    runpath_format = poly_run_model_config["runpath_config"]["runpath_format_string"]
    # Create realization 0, iter 0 directory
    runpath_0 = (
        Path(runpath_format % (0, 0))
        if "%d" in runpath_format
        else Path(runpath_format.replace("<IENS>", "0").replace("<ITER>", "0"))
    )
    runpath_0.mkdir(parents=True, exist_ok=True)

    response = experiment_server_client.post(
        check_runpath_url, json=poly_run_model_config, auth=_AUTH
    )
    assert response.status_code == 200
    data = response.json()
    assert data["runpath_exists"] is True


def test_that_check_runpath_requires_authentication(
    experiment_server_client, poly_run_model_config
):
    response = experiment_server_client.post(
        check_runpath_url, json=poly_run_model_config
    )
    assert response.status_code == 401


def test_that_check_runpath_returns_422_for_invalid_config(
    experiment_server_client,
):
    response = experiment_server_client.post(
        check_runpath_url, json={"type": "invalid_type"}, auth=_AUTH
    )
    assert response.status_code == 422


def test_that_delete_runpaths_removes_existing_run_directories(
    experiment_server_client, poly_run_model_config, tmp_path
):
    runpath_format = poly_run_model_config["runpath_config"]["runpath_format_string"]
    # Substitute template placeholders — use ERT's realization/iter format
    runpath_0 = Path(
        runpath_format.replace("<IENS>", "0").replace("<ITER>", "0")
        if "<IENS>" in runpath_format
        else runpath_format % (0, 0)
    )
    runpath_0.mkdir(parents=True, exist_ok=True)
    assert runpath_0.exists()

    response = experiment_server_client.post(
        _DELETE_RUNPATHS_URL, json=poly_run_model_config, auth=_AUTH
    )
    assert response.status_code == 200
    assert not runpath_0.exists()


def test_that_delete_runpaths_requires_authentication(
    experiment_server_client, poly_run_model_config
):
    response = experiment_server_client.post(
        _DELETE_RUNPATHS_URL, json=poly_run_model_config
    )
    assert response.status_code == 401


def test_that_has_failed_realizations_returns_false_when_no_realizations_failed(
    experiment_server_client,
):
    run_id = "test-run-no-failed"
    state = ExperimentRunnerState(has_failed_realizations=False)
    _runs[run_id] = state

    response = experiment_server_client.get(
        f"/experiment_server/{EverEndpoints.has_failed_realizations}/{run_id}",
        auth=_AUTH,
    )
    assert response.status_code == 200
    assert response.json() == {"has_failed": False}


def test_that_has_failed_realizations_returns_true_when_some_realizations_failed(
    experiment_server_client,
):
    run_id = "test-run-with-failed"
    state = ExperimentRunnerState(has_failed_realizations=True)
    _runs[run_id] = state

    response = experiment_server_client.get(
        f"/experiment_server/{EverEndpoints.has_failed_realizations}/{run_id}",
        auth=_AUTH,
    )
    assert response.status_code == 200
    assert response.json() == {"has_failed": True}


def test_that_has_failed_realizations_returns_404_for_unknown_run_id(
    experiment_server_client,
):
    response = experiment_server_client.get(
        f"/experiment_server/{EverEndpoints.has_failed_realizations}/nonexistent-id",
        auth=_AUTH,
    )
    assert response.status_code == 404


def test_that_rerun_failed_creates_new_run_with_same_model(
    experiment_server_client,
):
    mock_run_model = MagicMock()
    mock_run_model.supports_rerunning_failed_realizations = True

    run_id = "original-run-id"
    state = ExperimentRunnerState(
        run_model=mock_run_model,
        supports_rerunning_failed_realizations=True,
        config_path="/path/to/config.ert",
        run_path="/path/to/runpath",
        storage_path="/path/to/storage",
    )
    _runs[run_id] = state

    with patch(
        "ert.dark_storage.endpoints.experiment_server.ExperimentRunner.run"
    ) as mock_run:
        mock_run.return_value = None
        response = experiment_server_client.post(
            f"/experiment_server/{EverEndpoints.start_experiment}",
            params={"rerun_from_run_id": run_id},
            auth=_AUTH,
        )

    assert response.status_code == 200
    data = response.json()
    new_run_id = data["run_id"]
    assert new_run_id != run_id
    assert new_run_id in _runs
    assert _runs[new_run_id].run_model is mock_run_model
    assert data["supports_rerunning_failed_realizations"] is True


def test_that_rerun_failed_returns_400_when_run_model_is_missing(
    experiment_server_client,
):
    run_id = "run-without-model"
    state = ExperimentRunnerState(run_model=None)
    _runs[run_id] = state

    response = experiment_server_client.post(
        f"/experiment_server/{EverEndpoints.start_experiment}",
        params={"rerun_from_run_id": run_id},
        auth=_AUTH,
    )
    assert response.status_code == 400


def test_that_rerun_failed_returns_400_when_rerun_is_not_supported(
    experiment_server_client,
):
    mock_run_model = MagicMock()

    run_id = "run-no-rerun-support"
    state = ExperimentRunnerState(
        run_model=mock_run_model,
        supports_rerunning_failed_realizations=False,
    )
    _runs[run_id] = state

    response = experiment_server_client.post(
        f"/experiment_server/{EverEndpoints.start_experiment}",
        params={"rerun_from_run_id": run_id},
        auth=_AUTH,
    )
    assert response.status_code == 400


def test_that_rerun_failed_returns_404_for_unknown_run_id(
    experiment_server_client,
):
    response = experiment_server_client.post(
        f"/experiment_server/{EverEndpoints.start_experiment}",
        params={"rerun_from_run_id": "nonexistent-id"},
        auth=_AUTH,
    )
    assert response.status_code == 404
