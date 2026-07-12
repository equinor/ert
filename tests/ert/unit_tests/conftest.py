import io
import os
import sys
from io import BytesIO, StringIO

import numpy as np
import polars as pl
import pytest

from ert.config import ErtConfig
from ert.run_arg import RunArg, create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import Ensemble


@pytest.fixture(scope="session", autouse=True)
def ensure_bin_in_path():
    """
    Running pytest directly without enabling a virtualenv is perfectly valid.
    However, our tests assume that `fm_dispatch.py` is in PATH which it may not be.
    This fixture prepends the path to the current Python for tests to pass when not
    in a virtualenv.
    """
    path = os.environ["PATH"]
    exec_path = os.path.dirname(sys.executable)
    os.environ["PATH"] = exec_path + os.pathsep + path
    yield
    os.environ["PATH"] = path


original_open = open
original_io_open = io.open
original_stat = os.stat
original_pl_read_csv = pl.read_csv
original_np_loadtxt = np.loadtxt


@pytest.fixture
def mocked_files(mocker):
    mocked_files = {}

    def _fresh_buffer(data):
        if isinstance(data, bytes):
            return BytesIO(data)
        return StringIO(data)

    def mock_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        data = mocked_files.get(str(path))
        if data is not None:
            return _fresh_buffer(data)
        return original_open(*args, **kwargs)

    def mock_io_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        data = mocked_files.get(str(path))
        if data is not None:
            return _fresh_buffer(data)
        return original_io_open(*args, **kwargs)

    def mock_stat(*args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        if str(path) in mocked_files:
            return os.stat_result([0x777, *([1] * 10)])
        return original_stat(*args, **kwargs)

    def mock_pl_read_csv(*args, **kwargs):
        path = args[0] if args else kwargs.get("source")
        data = mocked_files.get(str(path))
        if data is not None:
            new_kwargs = {**kwargs}
            new_kwargs.pop("source", None)
            return original_pl_read_csv(_fresh_buffer(data), *args[1:], **new_kwargs)
        return original_pl_read_csv(*args, **kwargs)

    def mock_np_loadtxt(*args, **kwargs):
        path = args[0] if args else kwargs.get("fname")
        data = mocked_files.get(str(path))
        if data is not None:
            new_kwargs = {**kwargs}
            new_kwargs.pop("fname", None)
            return original_np_loadtxt(_fresh_buffer(data), *args[1:], **new_kwargs)
        return original_np_loadtxt(*args, **kwargs)

    mocker.patch("builtins.open", mock_open)
    mocker.patch("io.open", mock_io_open)
    mocker.patch("os.stat", mock_stat)
    mocker.patch("polars.read_csv", mock_pl_read_csv)
    mocker.patch("numpy.loadtxt", mock_np_loadtxt)

    return mocked_files


@pytest.fixture
def snake_oil_field_example(setup_case):
    return setup_case("snake_oil_field", "snake_oil_field.ert")


@pytest.fixture
def prior_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)


@pytest.fixture
def prior_ensemble_args(storage):
    def _create_prior_ensemble(
        ensemble_name="prior", ensemble_size=100, **experiment_params
    ):
        experiment_id = storage.create_experiment(**experiment_params)
        return storage.create_ensemble(
            experiment_id, name=ensemble_name, ensemble_size=ensemble_size
        )

    return _create_prior_ensemble


@pytest.fixture
def run_args():
    def func(
        ert_config: ErtConfig,
        ensemble: Ensemble,
        active_realizations: int | None = None,
    ) -> list[RunArg]:
        active_realizations = (
            ert_config.runpath_config.num_realizations
            if active_realizations is None
            else active_realizations
        )
        return create_run_arguments(
            Runpaths.from_config(ert_config),
            [True] * active_realizations,
            ensemble,
        )

    return func
