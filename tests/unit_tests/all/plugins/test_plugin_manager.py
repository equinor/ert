import logging
import tempfile
from unittest.mock import Mock

import ert.shared.hook_implementations
from ert.shared.plugins.plugin_manager import _ErtPluginManager
from tests.unit_tests.all.plugins import dummy_plugins


def test_no_plugins():
    pm = _ErtPluginManager(plugins=[ert.shared.hook_implementations])
    assert pm.get_help_links() == {"GitHub page": "https://github.com/equinor/ert"}
    assert pm.get_flow_config_path() is None
    assert pm.get_ecl100_config_path() is None
    assert pm.get_ecl300_config_path() is None
    assert pm.get_rms_config_path() is None

    assert len(pm.get_installable_jobs()) > 0
    assert len(pm._get_config_workflow_jobs()) > 0

    assert pm._site_config_lines() == [
        "-- Content below originated from ert (site_config_lines)",
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
    ]


def test_with_plugins():
    pm = _ErtPluginManager(plugins=[ert.shared.hook_implementations, dummy_plugins])
    assert pm.get_help_links() == {
        "GitHub page": "https://github.com/equinor/ert",
        "test": "test",
        "test2": "test",
    }
    assert pm.get_flow_config_path() == "/dummy/path/flow_config.yml"
    assert pm.get_rms_config_path() == "/dummy/path/rms_config.yml"
    assert pm.get_ecl100_config_path() == "/dummy/path/ecl100_config.yml"
    assert pm.get_ecl300_config_path() == "/dummy/path/ecl300_config.yml"

    assert pm.get_installable_jobs()["job1"] == "/dummy/path/job1"
    assert pm.get_installable_jobs()["job2"] == "/dummy/path/job2"
    assert pm._get_config_workflow_jobs()["wf_job1"] == "/dummy/path/wf_job1"
    assert pm._get_config_workflow_jobs()["wf_job2"] == "/dummy/path/wf_job2"

    assert pm._site_config_lines() == [
        "-- Content below originated from ert (site_config_lines)",
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "-- Content below originated from dummy (site_config_lines)",
        "JOB_SCRIPT job_dispatch_dummy.py",
        "QUEUE_OPTION LOCAL MAX_RUNNING 2",
    ]


def test_job_documentation():
    pm = _ErtPluginManager(plugins=[dummy_plugins])
    expected = {
        "job1": {
            "config_file": "/dummy/path/job1",
            "source_package": "dummy",
            "source_function_name": "installable_jobs",
            "description": "job description",
            "examples": "example 1 and example 2",
            "category": "test.category.for.job",
        },
        "job2": {
            "config_file": "/dummy/path/job2",
            "source_package": "dummy",
            "source_function_name": "installable_jobs",
        },
    }
    assert pm.get_documentation_for_jobs() == expected


def test_workflows_merge(monkeypatch, tmpdir):
    expected_result = {
        "wf_job1": "/dummy/path/wf_job1",
        "wf_job2": "/dummy/path/wf_job2",
        "some_func": str(tmpdir / "SOME_FUNC"),
    }
    tempfile_mock = Mock(return_value=tmpdir)
    monkeypatch.setattr(tempfile, "mkdtemp", tempfile_mock)
    pm = _ErtPluginManager(plugins=[dummy_plugins])
    result = pm.get_installable_workflow_jobs()
    assert result == expected_result


def test_workflows_merge_duplicate(caplog):
    pm = _ErtPluginManager(plugins=[dummy_plugins])

    dict_1 = {"some_job": "/a/path"}
    dict_2 = {"some_job": "/a/path"}

    with caplog.at_level(logging.INFO):
        result = pm._merge_internal_jobs(dict_1, dict_2)

    assert result == {"some_job": "/a/path"}

    assert (
        "Duplicate key: some_job in workflow hook implementations, "
        "config path 1: /a/path, config path 2: /a/path"
    ) in caplog.text


def test_add_logging_handle(tmpdir):
    with tmpdir.as_cwd():
        pm = _ErtPluginManager(plugins=[dummy_plugins])
        pm.add_logging_handle_to_root(logging.getLogger())
        logging.critical("I should write this to spam.log")
        with open("spam.log", encoding="utf-8") as fin:
            result = fin.read()
        assert "I should write this to spam.log" in result
