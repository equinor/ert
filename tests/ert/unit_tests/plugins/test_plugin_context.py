import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ert.config.queue_config import LsfQueueOptions
from ert.plugins import ErtPluginContext, ErtPluginManager, ErtRuntimePlugins
from tests.ert.unit_tests.plugins import dummy_plugins

env_vars = [
    "ECL100_SITE_CONFIG",
    "ECL300_SITE_CONFIG",
    "FLOW_SITE_CONFIG",
    "ERT_SITE_CONFIG",
]


class ErtPluginContextLegacy(ErtPluginContext):
    def __enter__(self) -> ErtPluginManager:
        """
        Hooks into the ErtPluginContext to get the actual ErtPluginManager.
        Background: There should be no direct interactions with ErtPluginManager.
        All interactions with the plugin env should be packed into a context,
        containing all relevant settings and configurations for ERT.
        """
        super().__enter__()
        return self.plugin_manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return super().__exit__(exc_type, exc_val, exc_tb)


def mock_dummy_plugins(target_dir: Path):
    fm_disapatch_path = target_dir / "fm_dispatch_dummy.py"
    fm_disapatch_path.write_text("echo helloworld")

    os.chmod(fm_disapatch_path, fm_disapatch_path.stat().st_mode | stat.S_IEXEC)

    Path.mkdir(target_dir / "dummy/path", exist_ok=True, parents=True)
    (target_dir / "dummy/path/job1").write_text("EXECUTABLE echo")
    (target_dir / "dummy/path/job2").write_text("EXECUTABLE echo")
    (target_dir / "dummy/path/wf_job1").write_text("echo job1")
    (target_dir / "dummy/path/wf_job2").write_text("echo job2")

    (target_dir / "TEST").write_text("")

    return dummy_plugins


def test_no_plugins(monkeypatch):
    monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
    site_config_dir = tempfile.mkdtemp()
    monkeypatch.setattr(tempfile, "mkdtemp", Mock(return_value=site_config_dir))
    monkeypatch.chdir(site_config_dir)

    with (
        ErtPluginContextLegacy(plugins=[]) as pm,
    ):
        with pytest.raises(KeyError):
            _ = os.environ["ECL100_SITE_CONFIG"]
        with pytest.raises(KeyError):
            _ = os.environ["ECL300_SITE_CONFIG"]
        with pytest.raises(KeyError):
            _ = os.environ["FLOW_SITE_CONFIG"]

        assert os.path.isfile(os.environ["ERT_SITE_CONFIG"])
        with open(os.environ["ERT_SITE_CONFIG"], encoding="utf-8") as f:
            assert pm.get_site_config_content() == f.read()

        path = os.environ["ERT_SITE_CONFIG"]

    with pytest.raises(KeyError):
        _ = os.environ["ERT_SITE_CONFIG"]
    assert not os.path.isfile(path)


def test_with_plugins(monkeypatch):
    monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
    # We are comparing two function calls, both of which generate a tmpdir,
    # this makes sure that the same tmpdir is called on both occasions.
    site_config_dir = tempfile.mkdtemp()
    monkeypatch.chdir(site_config_dir)
    mock_dummy_plugins(Path(site_config_dir))
    monkeypatch.setattr(tempfile, "mkdtemp", Mock(return_value=site_config_dir))

    with (
        ErtPluginContextLegacy(plugins=[dummy_plugins]) as pm,
    ):
        with pytest.raises(KeyError):
            _ = os.environ["ECL100_SITE_CONFIG"]
        with pytest.raises(KeyError):
            _ = os.environ["ECL300_SITE_CONFIG"]
        with pytest.raises(KeyError):
            _ = os.environ["FLOW_SITE_CONFIG"]

        assert os.path.isfile(os.environ["ERT_SITE_CONFIG"])
        with open(os.environ["ERT_SITE_CONFIG"], encoding="utf-8") as f:
            assert pm.get_site_config_content() == f.read()

        path = os.environ["ERT_SITE_CONFIG"]

    with pytest.raises(KeyError):
        _ = os.environ["ERT_SITE_CONFIG"]
    assert not os.path.isfile(path)


def test_already_set(monkeypatch):
    for var in env_vars:
        monkeypatch.setenv(var, "TEST")

    site_config_dir = tempfile.mkdtemp()
    monkeypatch.chdir(site_config_dir)
    mock_dummy_plugins(Path(site_config_dir))
    monkeypatch.setattr(tempfile, "mkdtemp", Mock(return_value=site_config_dir))

    with ErtPluginContextLegacy(plugins=[dummy_plugins]):
        for var in env_vars:
            assert os.environ[var] == "TEST"

    for var in env_vars:
        assert os.environ[var] == "TEST"


def test_that_site_configuration_propagates_through_plugin_manager():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    configs = pm.get_site_configurations()
    assert configs == ErtRuntimePlugins(
        queue_options=LsfQueueOptions(
            name="lsf",
            max_running="1",
            submit_sleep="1",
            job_script="fm_dispatch_sitecfg.py",
            activate_script="cminer",
        ),
        environment_variables={
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "NUMEXPR_NUM_THREADS": "5",
        },
    )
