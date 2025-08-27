import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(block_storage_path):
    ert_config = ErtConfig.from_file(
        str(block_storage_path / "version-3/poly_example/poly.ert")
    )
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


@pytest.mark.filterwarnings("ignore:.*The SIMULATION_JOB keyword has been removed")
def test_migrate_observations(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-3/poly_example", "poly.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = next(iter(storage.experiments))

        assert experiment.observations.keys() == ert_config.observations.keys()
        assert all(
            experiment.observations[k].equals(ert_config.observations[k])
            for k in experiment.observations
        )


@pytest.mark.filterwarnings("ignore:.*The SIMULATION_JOB keyword has been removed")
def test_migrate_gen_kw_config(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-3/poly_example", "poly.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = next(iter(storage.experiments))
        assert "template_file_path" not in experiment.parameter_configuration
