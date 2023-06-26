import logging
import re

import netCDF4
import numpy as np
import pytest
import xarray as xr
import xarray.backends

import ert.storage
import ert.storage.migration._block_fs_native as bfn
import ert.storage.migration.block_fs as bf
from ert._c_wrappers.enkf import ErtConfig
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture
def ensemble(storage):
    return storage.create_experiment().create_ensemble(name="default", ensemble_size=5)


@pytest.fixture(scope="module")
def enspath(block_storage_path):
    return block_storage_path / "simple_case/storage"


@pytest.fixture
def ert_config(block_storage_path, monkeypatch):
    monkeypatch.chdir(block_storage_path / "simple_case")
    return ErtConfig.from_file(str(block_storage_path / "simple_case/config.ert"))


@pytest.fixture
def ens_config(ert_config):
    return ert_config.ensemble_config


@pytest.fixture(autouse=True)
def set_ert_config(ert_config):
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


@pytest.fixture(scope="module")
def data(block_storage_path):
    return netCDF4.Dataset(block_storage_path / "data_dump/simple_case.nc")


@pytest.fixture(scope="module")
def forecast(enspath):
    return bfn.DataFile(enspath / "default/Ensemble/mod_0/FORECAST.data_0")


@pytest.fixture(scope="module")
def parameter(enspath):
    return bfn.DataFile(enspath / "default/Ensemble/mod_0/PARAMETER.data_0")


def sorted_surface(data):
    """Make sure that the data is sorted row-wise"""
    dataset = xr.open_dataset(xarray.backends.NetCDF4DataStore(data))
    return np.array(dataset.sortby(["Y_UTMN", "X_UTME"])["VALUES"]).ravel()


def test_migrate_surface(data, storage, parameter, ens_config):
    parameters = bf._migrate_surface_info(parameter, ens_config)
    experiment = storage.create_experiment(parameters=parameters)

    ensemble = experiment.create_ensemble(name="default", ensemble_size=5)
    bf._migrate_surface(ensemble, parameter, ens_config)

    for key, var in data["/REAL_0/SURFACE"].groups.items():
        expect = sorted_surface(var)
        actual = ensemble.load_parameters(key, 0).values.ravel()
        assert list(expect) == list(actual), key


def test_migrate_field(data, storage, parameter, ens_config):
    parameters = bf._migrate_field_info(parameter, ens_config)
    experiment = storage.create_experiment(parameters=parameters)

    ensemble = experiment.create_ensemble(name="default", ensemble_size=5)
    bf._migrate_field(ensemble, parameter, ens_config)

    for key, var in data["/REAL_0/FIELD"].groups.items():
        expect = np.array(var["VALUES"]).ravel()
        actual = ensemble.load_parameters(key, [0]).values.ravel()
        assert list(expect) == list(actual), key


def test_migrate_case(data, storage, enspath):
    bf.migrate_case(storage, enspath / "default")

    ensemble = storage.get_ensemble_by_name("default")
    for real_key, real_group in data.groups.items():
        real_index = int(re.match(r"REAL_(\d+)", real_key)[1])

        # Sanity check: Test data only contains FIELD and SURFACE
        assert set(real_group.groups) == {"FIELD", "SURFACE"}

        # Compare FIELDs
        for key, data in real_group["FIELD"].groups.items():
            expect = np.array(data["VALUES"]).ravel()
            actual = ensemble.load_parameters(key, [real_index])
            assert list(expect) == list(actual.values.ravel()), f"FIELD {key}"

        # Compare SURFACEs
        for key, data in real_group["SURFACE"].groups.items():
            expect = sorted_surface(data)
            actual = ensemble.load_parameters(key, real_index).values.ravel()
            assert list(expect) == list(actual), f"SURFACE {key}"


def test_migration_failure(storage, enspath, ens_config, caplog, monkeypatch):
    """Run migration but fail due to missing config data. Expected behaviour is
    for the error to be logged but no exception be propagated.

    """
    # Keep only the last parameter config
    param_to_keep = list(ens_config.parameter_configs)[-1]
    monkeypatch.setattr(
        ens_config, "parameter_configs", {param_to_keep: ens_config[param_to_keep]}
    )
    monkeypatch.setattr(ert.storage, "open_storage", lambda: storage)

    # Sanity check: no ensembles are created before migration
    assert list(storage.ensembles) == []

    with caplog.at_level(logging.WARNING, logger="ert.storage.migration.block_fs"):
        bf._migrate_case_ignoring_exceptions(storage, enspath / "default")

    # No ensembles were created due to failure
    assert list(storage.ensembles) == []

    # Warnings are in caplog
    assert len(caplog.records) == 1
    assert caplog.records[0].message == (
        "Exception occurred during migration of BlockFs case 'default': "
        "'The key:FIELD_1 is not in the ensemble configuration'"
    )
