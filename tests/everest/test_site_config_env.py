import os
import shutil
from unittest.mock import patch

import pytest

from everest.plugins import hook_impl as everest_implementation
from everest.plugins import hookimpl
from everest.plugins.everest_plugin_manager import EverestPluginManager
from everest.plugins.plugin_response import plugin_response
from everest.plugins.site_config_env import PluginSiteConfigEnv


class DummyPlugin2:
    @hookimpl
    @plugin_response(plugin_name="Dummy2")  # pylint: disable=no-value-for-parameter
    def site_config_lines(self):
        return ["-- dummy site config from plugin 2", ""]

    @hookimpl
    @plugin_response(plugin_name="Dummy2")  # pylint: disable=no-value-for-parameter
    def installable_workflow_jobs(self):
        return {"dummy_job": "dummy/workflow/job/path"}


class DummyPlugin:
    @hookimpl
    @plugin_response(plugin_name="Dummy")  # pylint: disable=no-value-for-parameter
    def site_config_lines(self):
        return ["-- dummy site config", ""]

    @hookimpl
    @plugin_response(plugin_name="Dummy")  # pylint: disable=no-value-for-parameter
    def ecl100_config_path(self):
        return "dummy/ecl100_config_path"

    @hookimpl
    @plugin_response(plugin_name="Dummy")  # pylint: disable=no-value-for-parameter
    def ecl300_config_path(self):
        return "dummy/ecl300_config_path"

    @hookimpl
    @plugin_response(plugin_name="Dummy")  # pylint: disable=no-value-for-parameter
    def flow_config_path(self):
        return "dummy/flow_config_path"


@pytest.fixture
def mocked_env(mocker):
    @patch(
        "everest.plugins.site_config_env.EverestPluginManager",
        return_value=EverestPluginManager(
            [DummyPlugin(), DummyPlugin2(), everest_implementation]
        ),
    )
    def get_env(*args):
        return PluginSiteConfigEnv()

    return get_env()


expected_env_lines = [
    "SETENV ECL100_SITE_CONFIG dummy/ecl100_config_path",
    "SETENV ECL300_SITE_CONFIG dummy/ecl300_config_path",
    "SETENV FLOW_SITE_CONFIG dummy/flow_config_path",
    "",
]

expected_site_config_extra = [
    "JOB_SCRIPT job_dispatch.py",
    "QUEUE_OPTION LOCAL MAX_RUNNING 1",
    "",
]

expected_site_config_content = (
    "\n".join(
        [
            *expected_site_config_extra,
            "-- dummy site config",
            "",
            "-- dummy site config from plugin 2",
            "",
            *expected_env_lines,
            "LOAD_WORKFLOW_JOB dummy/workflow/job/path",
            "",
        ]
    )
    + "\n"
)


def test_add_config_env_vars(mocked_env):
    result = mocked_env._config_env_vars()
    assert result == expected_env_lines


def test_get_temp_site_config_path(mocked_env):
    result = mocked_env._get_temp_site_config_path()
    expected = os.path.join(mocked_env.tmp_dir, "site-config")
    assert result == expected
    if mocked_env.tmp_dir is not None:
        shutil.rmtree(mocked_env.tmp_dir)


def test_get_site_config_content(mocked_env):
    result = mocked_env._get_site_config_content()
    assert result == expected_site_config_content


def test_write_tmp_site_config_file(tmpdir, mocked_env):
    with tmpdir.as_cwd():
        site_conf_path = "test-site-config"
        assert not os.path.exists(site_conf_path)
        mocked_env._write_tmp_site_config_file(
            path=site_conf_path, content="test content"
        )
        assert os.path.exists("test-site-config")
        with open(site_conf_path, "r", encoding="utf-8") as f:
            assert f.read() == "test content"


def test_env_context(mocked_env):
    assert os.environ.get("ERT_SITE_CONFIG", "NOT_SET") == "NOT_SET"
    with mocked_env:
        site_config_path = os.path.join(mocked_env.tmp_dir, "site-config")
        assert os.environ.get("ERT_SITE_CONFIG", "NOT_SET") == site_config_path
        os.path.exists(site_config_path)
        with open(site_config_path, "r", encoding="utf-8") as f:
            assert expected_site_config_content == f.read()
    assert not os.path.exists(mocked_env.tmp_dir)
    assert os.environ.get("ERT_SITE_CONFIG", "NOT_SET") == "NOT_SET"
