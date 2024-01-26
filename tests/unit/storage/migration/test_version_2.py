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


def test_migrate_gen_kw(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = list(storage.experiments)[0]
        response_info = json.loads(
            (experiment._path / "responses.json").read_text(encoding="utf-8")
        )
        assert (
            list(experiment.response_configuration.values())
            == ert_config.ensemble_config.response_configuration
        )
    assert list(response_info) == [
        "SNAKE_OIL_OPR_DIFF",
        "SNAKE_OIL_WPR_DIFF",
        "SNAKE_OIL_GPR_DIFF",
        "summary",
    ]


def test_migrate_gen_kw_config(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = list(storage.experiments)[0]
        assert "template_file_path" not in experiment.parameter_configuration
