from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from ert.plugins import ErtPluginManager
from ert.plugins.hook_implementations.workflows.export_runpath import ExportRunpathJob
from ert.runpaths import Runpaths


@pytest.fixture
def snake_oil_export_runpath_job(setup_case):
    setup_case("snake_oil", "snake_oil.ert")
    plugin = ExportRunpathJob()
    return plugin


@dataclass
class WritingSetup:
    write_mock: Mock
    export_job: ExportRunpathJob


@pytest.fixture
def writing_setup(setup_case):
    with patch.object(Runpaths, "write_runpath_list") as write_mock:
        config = setup_case("snake_oil", "snake_oil.ert")
        yield (
            WritingSetup(
                write_mock,
                ExportRunpathJob(),
            ),
            Runpaths.from_config(config),
        )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_empty_range(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_ensemble.ensemble_size = 25
    writing_setup.export_job.run(run_paths, mock_ensemble, [])

    writing_setup.write_mock.assert_called_with(
        [0], list(range(mock_ensemble.ensemble_size))
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_star_parameter(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 1
    mock_ensemble.ensemble_size = 25
    writing_setup.export_job.run(run_paths, mock_ensemble, ["* | *"])

    writing_setup.write_mock.assert_called_with(
        list(range(1)),
        list(range(25)),
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_range_parameter(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_ensemble.ensemble_size = 25
    writing_setup.export_job.run(run_paths, mock_ensemble, ["* | 1-2"])

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        list(range(25)),
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_comma_parameter(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_ensemble.ensemble_size = 25
    writing_setup.export_job.run(run_paths, mock_ensemble, ["3,4 | 1-2"])

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        [3, 4],
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_combination_parameter(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_ensemble.ensemble_size = 25
    writing_setup.export_job.run(run_paths, mock_ensemble, ["1,2-3 | 1-2"])

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        [1, 2, 3],
    )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_export_runpath_bad_arguments(writing_setup):
    writing_setup, run_paths = writing_setup

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_ensemble.ensemble_size = 25
    with pytest.raises(ValueError, match="Expected \\|"):
        writing_setup.export_job.run(run_paths, mock_ensemble, ["wat"])


def test_export_runpath_job_is_loaded():
    pm = ErtPluginManager()
    assert "EXPORT_RUNPATH" in pm.get_ertscript_workflows().get_workflows()
