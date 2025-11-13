import pytest

from ert.config import ErtConfig
from ert.dark_storage.endpoints.parameters import data_for_parameter
from ert.storage import open_storage


@pytest.mark.integration_test
def test_that_asking_for_non_existent_key_doesnt_raise(
    symlinked_heat_equation_storage_es,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        ensemble = next(storage.ensembles)
        assert "variable" not in ensemble.experiment.parameter_configuration
        df = data_for_parameter(ensemble, "RANDOM:variable")
        assert df.empty


@pytest.mark.integration_test
def test_that_asking_for_existing_key_without_group_returns_empty(
    symlinked_heat_equation_storage_es,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        ensemble = next(storage.ensembles)
        assert ensemble.experiment.parameter_configuration["COND"]
        df = data_for_parameter(ensemble, "COND")
        assert df.empty


@pytest.mark.integration_test
def test_that_asking_for_existing_key_with_group_returns_data(
    symlinked_heat_equation_storage_es,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        ensemble = next(storage.ensembles)
        assert "t" in ensemble.experiment.parameter_configuration
        df = data_for_parameter(ensemble, "INIT_TEMP_SCALE:t")
        assert not df.empty
