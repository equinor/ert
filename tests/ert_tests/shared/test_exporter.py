import shutil
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

import ert_shared.libres_facade
from ert_shared.exporter import Exporter
from ert_shared.plugins.plugin_manager import ErtPluginContext
from res.enkf import EnKFMain, ResConfig


@pytest.fixture(autouse=True)
def snake_oil_data(source_root, tmp_path, monkeypatch):
    shutil.copytree(
        source_root / "test-data/local/snake_oil", tmp_path, dirs_exist_ok=True
    )
    monkeypatch.chdir(tmp_path)


def test_exporter_is_valid():
    with ErtPluginContext():
        config_file = "snake_oil.ert"
        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        ex = Exporter(ert)
        assert ex.is_valid(), "Missing CSV_EXPORT2 or EXPORT_RUNPATH jobs"


def test_exporter_is_not_valid():
    config_file = "snake_oil.ert"
    rc = ResConfig(user_config_file=config_file)
    rc.convertToCReference(None)
    ert = EnKFMain(rc)
    ex = Exporter(ert)
    assert not ex.is_valid()


def test_run_export():
    with ErtPluginContext():
        config_file = "snake_oil.ert"
        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        ex = Exporter(ert)
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


def test_run_export_pathfile(monkeypatch):
    with ErtPluginContext():
        run_path_file = Path("output/run_path_file/.some_new_name")
        config_file = "snake_oil.ert"
        with open(config_file, encoding="utf-8", mode="a") as fout:
            fout.write(f"RUNPATH_FILE {run_path_file}\n")
        rc = ResConfig(user_config_file=config_file)
        ert = EnKFMain(rc)
        run_mock = MagicMock()
        run_mock.hasFailed.return_value = False
        export_mock = MagicMock()
        export_mock.hasFailed.return_value = False

        monkeypatch.setattr(
            ert_shared.exporter.LibresFacade,
            "get_workflow_job",
            MagicMock(side_effect=[export_mock, run_mock]),
        )
        ex = Exporter(ert)
        parameters = {
            "output_file": "export.csv",
            "time_index": "raw",
            "column_keys": "FOPR",
        }
        ex.run_export(parameters)
        expected_call = call(
            arguments=[f"{run_path_file.absolute()}", "export.csv", "raw", "FOPR"],
            ert=ert,
            verbose=True,
        )
        assert export_mock.run.call_args == expected_call
