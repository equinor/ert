import os
import pytest
import shutil
import sys
from ert_shared.exporter import Exporter
from utils import SOURCE_DIR, tmpdir
from res.enkf import ResConfig, EnKFMain
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared import ERT
from ert_shared.plugins.plugin_manager import ErtPluginContext


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
@tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
def test_exporter_is_valid():
    with ErtPluginContext():
        config_file = "snake_oil.ert"
        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        notifier = ErtCliNotifier(ert, config_file)
        ERT.adapt(notifier)
        ex = Exporter()
        assert ex.is_valid(), "Missing CSV_EXPORT2 or EXPORT_RUNPATH jobs"


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
@tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
def test_exporter_is_not_valid():
    config_file = "snake_oil.ert"
    rc = ResConfig(user_config_file=config_file)
    rc.convertToCReference(None)
    ert = EnKFMain(rc)
    notifier = ErtCliNotifier(ert, config_file)
    ERT.adapt(notifier)
    ex = Exporter()
    assert not ex.is_valid()


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
@tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
def test_run_export():
    with ErtPluginContext():
        config_file = "snake_oil.ert"
        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        notifier = ErtCliNotifier(ert, config_file)
        ERT.adapt(notifier)
        ex = Exporter()
        parameters = {
            "output_file": "export.csv",
            "time_index": "raw",
            "column_keys": "FOPR",
        }
        ex.run_export(parameters)

        shutil.rmtree("storage")
        with pytest.raises(UserWarning) as warn:
            ex.run_export(parameters)
        assert ex._export_job in str(warn)
