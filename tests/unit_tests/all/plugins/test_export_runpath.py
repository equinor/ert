from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from ert.enkf_main import EnKFMain
from ert.runpaths import Runpaths
from ert.shared.hook_implementations.workflows.export_runpath import ExportRunpathJob
from ert.shared.plugins import ErtPluginManager
from storage import open_storage


@pytest.fixture
def snake_oil_export_runpath_job(setup_case):
    ert_config = setup_case("snake_oil", "snake_oil.ert")
    ert = EnKFMain(ert_config)
    with open_storage(ert.ert_config.ens_path, mode="w") as storage:
        yield ExportRunpathJob(ert, storage)


def test_export_runpath_number_of_realizations(snake_oil_export_runpath_job):
    assert snake_oil_export_runpath_job.number_of_realizations == 25


def test_export_runpath_number_of_iterations(snake_oil_export_runpath_job):
    assert snake_oil_export_runpath_job.number_of_iterations == 4


@dataclass
class WritingSetup:
    write_mock: Mock
    export_job: ExportRunpathJob


@pytest.fixture
def writing_setup(setup_case):
    with patch.object(Runpaths, "write_runpath_list") as write_mock:
        ert_config = setup_case("snake_oil", "snake_oil.ert")
        ert = EnKFMain(ert_config)
        yield WritingSetup(write_mock, ExportRunpathJob(ert, None))


def test_export_runpath_no_parameters(writing_setup):
    writing_setup.export_job.run()

    writing_setup.write_mock.assert_called_with(
        [0],
        list(range(writing_setup.export_job.number_of_realizations)),
    )


def test_export_runpath_star_parameter(writing_setup):
    writing_setup.export_job.run("* | *")

    writing_setup.write_mock.assert_called_with(
        list(range(writing_setup.export_job.number_of_iterations)),
        list(range(writing_setup.export_job.number_of_realizations)),
    )


def test_export_runpath_range_parameter(writing_setup):
    writing_setup.export_job.run("* | 1-2")

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        list(range(writing_setup.export_job.number_of_realizations)),
    )


def test_export_runpath_comma_parameter(writing_setup):
    writing_setup.export_job.run("3,4 | 1-2")

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        [3, 4],
    )


def test_export_runpath_combination_parameter(writing_setup):
    writing_setup.export_job.run("1,2-3 | 1-2")

    writing_setup.write_mock.assert_called_with(
        [1, 2],
        [1, 2, 3],
    )


def test_export_runpath_bad_arguments(writing_setup):
    with pytest.raises(ValueError, match="Expected |"):
        writing_setup.export_job.run("wat")


def test_export_runpath_job_is_loaded():
    pm = ErtPluginManager()
    assert "EXPORT_RUNPATH" in pm.get_installable_workflow_jobs()
