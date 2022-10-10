import pytest

from ert._c_wrappers.enkf import EnKFMain


@pytest.fixture()
def snake_oil_field_example(setup_case):
    return EnKFMain(setup_case("snake_oil_field", "snake_oil_field.ert"))
