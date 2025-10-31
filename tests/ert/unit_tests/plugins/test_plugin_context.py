import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ert import ForwardModelStepPlugin
from ert.config.queue_config import LsfQueueOptions
from ert.plugins import ErtPluginManager, ErtRuntimePlugins, get_site_plugins, plugin
from tests.ert.unit_tests.plugins import dummy_plugins


def mock_dummy_plugins(target_dir: Path):
    fm_disapatch_path = target_dir / "fm_dispatch_dummy.py"
    fm_disapatch_path.write_text("echo helloworld")

    os.chmod(fm_disapatch_path, fm_disapatch_path.stat().st_mode | stat.S_IEXEC)

    Path.mkdir(target_dir / "dummy/path", exist_ok=True, parents=True)
    executable_path = target_dir / "dummy/path/dummy_exec.sh"

    executable_path.write_text("#!/usr/bin/env bash\necho 'hello world'")
    os.chmod(executable_path, 0o775)

    (target_dir / "dummy/path/job1").write_text("EXECUTABLE dummy_exec.sh")
    (target_dir / "dummy/path/job2").write_text("EXECUTABLE dummy_exec.sh")
    (target_dir / "dummy/path/wf_job1").write_text("EXECUTABLE dummy_exec.sh")
    (target_dir / "dummy/path/wf_job2").write_text("EXECUTABLE dummy_exec.sh")

    (target_dir / "TEST").write_text("")

    return dummy_plugins


def test_no_plugins():
    runtime_plugins = get_site_plugins(ErtPluginManager(plugins=[]))
    assert ErtRuntimePlugins(queue_options=None) == runtime_plugins


def test_that_ecl_and_flow_envvars_plugins_are_passed_through_plugin_context(
    monkeypatch,
):
    some_tmpdir = tempfile.mkdtemp()
    monkeypatch.chdir(some_tmpdir)
    mock_dummy_plugins(Path(some_tmpdir))
    monkeypatch.setattr(tempfile, "mkdtemp", Mock(return_value=some_tmpdir))

    runtime_plugins = get_site_plugins(
        plugin_manager=ErtPluginManager(plugins=[dummy_plugins])
    )
    assert runtime_plugins.environment_variables == {
        "ECL100_SITE_CONFIG": "dummy/path/ecl100_config.yml",
        "ECL300_SITE_CONFIG": "dummy/path/ecl300_config.yml",
        "FLOW_SITE_CONFIG": "dummy/path/flow_config.yml",
        "OMP_NUM_THREADS": "5",
        "MKL_NUM_THREADS": "5",
        "NUMEXPR_NUM_THREADS": "5",
    }


def test_that_site_configuration_propagates_through_plugin_manager():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    configs = pm.get_site_configurations()
    assert configs == ErtRuntimePlugins(
        queue_options=LsfQueueOptions(
            name="lsf",
            max_running="1",
            submit_sleep="1",
        ),
        environment_variables={
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "NUMEXPR_NUM_THREADS": "5",
        },
    )


def test_that_site_configuration_propagates_through_plugin_context():
    site_config_content = {
        "queue_options": {
            "name": "lsf",
            "max_running": "1",
            "submit_sleep": "1",
        },
        "environment_variables": {
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "NUMEXPR_NUM_THREADS": "5",
        },
    }

    class SomePlugin:
        @plugin(name="some_dummy")
        def site_configurations():
            return site_config_content

    runtime_plugins = get_site_plugins(ErtPluginManager(plugins=[SomePlugin]))
    assert runtime_plugins == ErtRuntimePlugins(**site_config_content)


def test_that_site_configuration_forward_models_are_merged_with_other_plugins():
    class OPM9000(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="OPM9000",
                command=[
                    "opm9k",
                    "<ECLBASE>",
                    "--version",
                    "<VERSION>",
                    "-n",
                    "<NUM_CPU>",
                    "<OPTS>",
                ],
                default_mapping={
                    "<NUM_CPU>": "1",
                    "<OPTS>": "",
                    "<VERSION>": "version",
                },
            )

    class OPM10000(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(
                name="OPM10000",
                command=[
                    "opm10k",
                    "<ECLBASE>",
                    "--version",
                    "<VERSION>",
                    "-n",
                    "<NUM_CPU>",
                    "<OPTS>",
                ],
                default_mapping={
                    "<NUM_CPU>": "1",
                    "<OPTS>": "",
                    "<VERSION>": "version",
                },
            )

    site_config_content = {
        "installed_forward_model_steps": {OPM9000().name: OPM9000()},
        "environment_variables": {
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "NUMEXPR_NUM_THREADS": "5",
        },
    }

    class SomePlugin:
        @plugin(name="some_dummy")
        def site_configurations():
            return site_config_content

        @plugin(name="some_dummy")
        def installable_forward_model_steps():
            return [OPM10000]

    runtime_plugins = get_site_plugins(ErtPluginManager(plugins=[SomePlugin]))
    assert runtime_plugins.installed_forward_model_steps == {
        "OPM9000": OPM9000(),
        "OPM10000": OPM10000(),
    }


def test_that_plugin_context_with_two_site_configurations_raises_error(tmpdir):
    class SiteOne:
        @plugin(name="foo")
        def site_configurations():
            return ErtRuntimePlugins(environment_variables={"a": "b"})

    class SiteTwo:
        @plugin(name="foo")
        def site_configurations():
            return ErtRuntimePlugins(environment_variables={"a": "c"})

    with (
        pytest.raises(ValueError, match="Only one site configuration is allowed"),
    ):
        get_site_plugins(ErtPluginManager(plugins=[SiteOne, SiteTwo]))
