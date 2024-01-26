import json

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(block_storage_path):
    ert_config = ErtConfig.from_file(
        str(block_storage_path / "version-1/poly_example/poly.ert")
    )
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


def test_migrate_gen_kw(setup_case, set_ert_config):
    setup_case("block_storage/version-1/poly_example", "poly.ert")
    with open_storage("storage", "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = list(storage.experiments)[0]
        param_info = json.loads(
            (experiment._path / "parameter.json").read_text(encoding="utf-8")
        )
    assert "COEFFS" in param_info
