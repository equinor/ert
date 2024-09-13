import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(block_storage_path):
    ert_config = ErtConfig.from_file(
        str(block_storage_path / "version-4/no_summary_case/config.ert")
    )
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


def test_summary_config(block_storage_path, setup_case):
    ert_config = setup_case(
        str(block_storage_path / "version-4/no_summary_case/"), "config.ert"
    )
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = next(iter(storage.experiments))
        assert len(experiment.response_configuration) == 0


def test_metadata(block_storage_path, setup_case):
    ert_config = setup_case(
        str(block_storage_path / "version-4/no_summary_case/"), "config.ert"
    )
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = next(iter(storage.experiments))
        assert experiment.metadata == {}
