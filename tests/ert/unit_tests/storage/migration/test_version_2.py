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


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
@pytest.mark.filterwarnings("ignore:IES_ENKF has been removed and has no effect")
def test_migrate_responses(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = next(iter(storage.experiments))
        response_info = json.loads(
            (experiment._path / "responses.json").read_text(encoding="utf-8")
        )

        response_config_exp = experiment.response_configuration
        response_config_ens = ert_config.ensemble_config.response_configs

        # From storage v9 and onwards the response config is mutated
        # when migrating an existing experiment, because we check that the
        # keys in response.json are aligned with the dataset.
        response_config_ens["summary"].has_finalized_keys = response_config_exp[
            "summary"
        ].has_finalized_keys
        response_config_ens["summary"].keys = response_config_exp["summary"].keys

        assert response_config_exp == response_config_ens

    assert set(response_info) == {
        "gen_data",
        "summary",
    }


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
@pytest.mark.filterwarnings("ignore:IES_ENKF has been removed and has no effect")
def test_migrate_gen_kw_config(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-2/snake_oil", "snake_oil.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = next(iter(storage.experiments))
        assert "template_file_path" not in experiment.parameter_configuration
