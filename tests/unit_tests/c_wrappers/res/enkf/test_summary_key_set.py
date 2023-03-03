import os

import pytest

from ert._c_wrappers.enkf import EnKFMain, ErtConfig, SummaryKeySet
from ert._c_wrappers.enkf.enkf_fs import EnkfFs


def test_creation():
    keys = SummaryKeySet()

    assert len(keys) == 0

    assert keys.addSummaryKey("FOPT")

    assert len(keys) == 1

    assert "FOPT" in keys

    assert ["FOPT"] == list(keys.keys())

    assert keys.addSummaryKey("WWCT")

    assert len(keys), 2

    assert "WWCT" in keys

    assert list(keys.keys()) == ["FOPT", "WWCT"]


@pytest.mark.usefixtures("use_tmpdir")
def test_read_only_creation():
    keys = SummaryKeySet()

    keys.addSummaryKey("FOPT")
    keys.addSummaryKey("WWCT")

    filename = "test.txt"
    keys.writeToFile(filename)

    keys_from_file = SummaryKeySet(filename, read_only=True)
    assert keys.keys() == keys_from_file.keys()

    assert keys_from_file.isReadOnly()
    assert not keys_from_file.addSummaryKey("WOPR")


def test_write_to_and_read_from_file(tmp_path):
    keys = SummaryKeySet()

    keys.addSummaryKey("FOPT")
    keys.addSummaryKey("WWCT")

    filename = tmp_path / "test.txt"

    assert not os.path.exists(filename)

    keys.writeToFile(str(filename))

    assert os.path.exists(filename)

    keys_from_file = SummaryKeySet(str(filename))
    assert keys.keys() == keys_from_file.keys()


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_with_enkf_fs():
    ert_config = ErtConfig.from_file("snake_oil.ert")

    fs = EnkfFs(
        "storage/snake_oil/ensemble/default_0", ert_config.ensemble_config, 4, False
    )
    summary_key_set = fs._summary_key_set().setParent(fs)
    summary_key_set.addSummaryKey("FOPT")
    summary_key_set.addSummaryKey("WWCT")
    summary_key_set.addSummaryKey("WOPR")
    fs.sync()

    ert = EnKFMain(ert_config)
    fs = ert.storage_manager.current_case
    summary_key_set = fs.getSummaryKeySet()
    assert "FOPT" in summary_key_set
    assert "WWCT" in summary_key_set
    assert "WOPR" in summary_key_set

    ensemble_config = ert.ensembleConfig()

    assert "FOPT" in ensemble_config
    assert "TCPU" not in ensemble_config
