import os
import sys
from typing import List, Optional

import pytest

from ert.config import ErtConfig
from ert.run_arg import RunArg, create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import Ensemble


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


@pytest.fixture
def run_paths():
    def func(ert_config: ErtConfig):
        return Runpaths(
            jobname_format=ert_config.model_config.jobname_format_string,
            runpath_format=ert_config.model_config.runpath_format_string,
            filename=str(ert_config.runpath_file),
            substitution_list=ert_config.substitution_list,
        )

    yield func


@pytest.fixture
def run_args(run_paths):
    def func(
        ert_config: ErtConfig,
        ensemble: Ensemble,
        active_realizations: Optional[int] = None,
    ) -> List[RunArg]:
        active_realizations = (
            ert_config.model_config.num_realizations
            if active_realizations is None
            else active_realizations
        )
        return create_run_arguments(
            run_paths(ert_config),
            [True] * active_realizations,
            ensemble,
        )

    yield func
