import pytest

from ert._c_wrappers.enkf import EnKFMain


@pytest.fixture()
def snake_oil_field_example(setup_case):
    return EnKFMain(setup_case("snake_oil_field", "snake_oil_field.ert"))


@pytest.fixture
def prior_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)
