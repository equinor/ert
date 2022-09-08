import pytest

from ert._c_wrappers.enkf import EnKFMain


@pytest.fixture()
def snake_oil_example(setup_case):
    return EnKFMain(setup_case("local/snake_oil", "snake_oil.ert"))


@pytest.fixture()
def snake_oil_field_example(setup_case):
    return EnKFMain(setup_case("local/snake_oil_field", "snake_oil.ert"))
