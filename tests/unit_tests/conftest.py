import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_bin_in_path():
    """
    Running pytest directly without enabling a virtualenv is perfectly valid.
    However, our tests assume that `job_dispatch.py` is in PATH which it may not be.
    This fixture prepends the path to the current Python for tests to pass when not
    in a virtualenv.
    """
    path = os.environ["PATH"]
    exec_path = os.path.dirname(sys.executable)
    os.environ["PATH"] = exec_path + ":" + path


@pytest.fixture()
def snake_oil_field_example(setup_case):
    return setup_case("snake_oil_field", "snake_oil_field.ert")


@pytest.fixture
def prior_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)
