import json

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(block_storage_path):
    ert_config = ErtConfig.from_file(
        str(block_storage_path / "version-2/snake_oil/snake_oil.ert")
    )
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


def test_migrate_responses(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = next(iter(storage.experiments))
        response_info = json.loads(
            (experiment._path / "responses.json").read_text(encoding="utf-8")
        )
        assert (
            experiment.response_configuration
            == ert_config.ensemble_config.response_configs
        )

    assert set(response_info) == {
        "gen_data",
        "summary",
    }


def test_migrate_gen_kw_config(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = next(iter(storage.experiments))
        assert "template_file_path" not in experiment.parameter_configuration
