# -*- coding: utf-8 -*-
import sys
import unittest
import logging
from os import uname
import tempfile

import pytest
from ert_shared.plugins import ErtPluginManager
import ert_shared.hook_implementations

import tests.all.plugins.dummy_plugins as dummy_plugins

_lib_extension = "dylib" if uname()[0] == "Darwin" else "so"
if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


class PluginManagerTest(unittest.TestCase):
    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_no_plugins(self):
        pm = ErtPluginManager(plugins=[ert_shared.hook_implementations])
        self.assertDictEqual(
            {"GitHub page": "https://github.com/equinor/ert"}, pm.get_help_links()
        )
        self.assertIsNone(pm.get_flow_config_path())
        self.assertIsNone(pm.get_ecl100_config_path())
        self.assertIsNone(pm.get_ecl300_config_path())
        self.assertIsNone(pm.get_rms_config_path())

        self.assertLess(0, len(pm.get_installable_jobs()))
        self.assertLess(0, len(pm._get_installable_workflow_jobs()))

        self.assertListEqual(
            [
                "-- Content below originated from ert (site_config_lines)",
                "JOB_SCRIPT job_dispatch.py",
                "QUEUE_OPTION LOCAL MAX_RUNNING 1",
                "ANALYSIS_LOAD RML_ENKF rml_enkf.{}".format(_lib_extension),
            ],
            pm._site_config_lines(),
        )

    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_with_plugins(self):
        pm = ErtPluginManager(plugins=[ert_shared.hook_implementations, dummy_plugins])
        self.assertDictEqual(
            {
                "GitHub page": "https://github.com/equinor/ert",
                "test": "test",
                "test2": "test",
            },
            pm.get_help_links(),
        )
        self.assertEqual("/dummy/path/flow_config.yml", pm.get_flow_config_path())
        self.assertEqual("/dummy/path/rms_config.yml", pm.get_rms_config_path())
        self.assertEqual("/dummy/path/ecl100_config.yml", pm.get_ecl100_config_path())
        self.assertEqual("/dummy/path/ecl300_config.yml", pm.get_ecl300_config_path())

        self.assertIn(("job1", "/dummy/path/job1"), pm.get_installable_jobs().items())
        self.assertIn(("job2", "/dummy/path/job2"), pm.get_installable_jobs().items())
        self.assertIn(
            ("wf_job1", "/dummy/path/wf_job1"),
            pm._get_installable_workflow_jobs().items(),
        )
        self.assertIn(
            ("wf_job2", "/dummy/path/wf_job2"),
            pm._get_installable_workflow_jobs().items(),
        )

        self.assertListEqual(
            [
                "-- Content below originated from ert (site_config_lines)",
                "JOB_SCRIPT job_dispatch.py",
                "QUEUE_OPTION LOCAL MAX_RUNNING 1",
                "ANALYSIS_LOAD RML_ENKF rml_enkf.{}".format(_lib_extension),
                "-- Content below originated from dummy (site_config_lines)",
                "JOB_SCRIPT job_dispatch_dummy.py",
                "QUEUE_OPTION LOCAL MAX_RUNNING 2",
            ],
            pm._site_config_lines(),
        )

    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_job_documentation(self):
        pm = ErtPluginManager(plugins=[dummy_plugins])
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

    @unittest.skipIf(
        sys.version_info.major > 2, "Skipping Plugin Manager Python 2 test"
    )
    def test_plugin_manager_python_2(self):
        pm = ErtPluginManager()
        self.assertEqual(pm._get_installable_workflow_jobs(), None)
        self.assertEqual(pm.get_installable_jobs(), None)
        self.assertEqual(pm.get_flow_config_path(), None)
        self.assertEqual(pm.get_ecl100_config_path(), None)
        self.assertEqual(pm.get_ecl300_config_path(), None)
        self.assertEqual(pm.get_rms_config_path(), None)
        self.assertEqual(pm.get_help_links(), None)
        self.assertEqual(pm.get_site_config_content(), None)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_workflows_merge(monkeypatch, tmpdir):
    expected_result = {
        "wf_job1": "/dummy/path/wf_job1",
        "wf_job2": "/dummy/path/wf_job2",
        "some_func": str(tmpdir / "SOME_FUNC"),
    }
    tempfile_mock = Mock(return_value=tmpdir)
    monkeypatch.setattr(tempfile, "mkdtemp", tempfile_mock)
    pm = ErtPluginManager(plugins=[dummy_plugins])
    result = pm._get_workflow_jobs()
    assert result == expected_result


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_workflows_merge_duplicate(caplog):
    pm = ErtPluginManager(plugins=[dummy_plugins])

    dict_1 = {"some_job": "/a/path"}
    dict_2 = {"some_job": "/a/path"}

    with caplog.at_level(logging.INFO):
        result = pm._merge_internal_jobs(dict_1, dict_2)

    assert result == {"some_job": "/a/path"}

    assert (
        "Duplicate key: some_job in workflow hook implementations, config path 1: /a/path, config path 2: /a/path"
        in caplog.text
    )
