import pytest
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
import os


def test_storage_driver(tmpdir):
    storage_path = ".my_storage_test_path"
    test_config = {"type": "shared_disk", "storage_path": storage_path}

    with tmpdir.as_cwd():
        file_name = "test_file.txt"
        with open(file_name, "w") as f:
            f.write("...")

        storage = storage_driver_factory(config=test_config, run_path=tmpdir.strpath)

        assert storage is not None

        # Check storage path
        global_storage_path = storage.get_storage_path(None)
        expected_global_storage_path = os.path.join(storage_path, "global")
        assert expected_global_storage_path == global_storage_path

        storage_path_iens_1 = storage.get_storage_path(1)
        expected_storage_path = os.path.join(storage_path, "1")
        assert expected_storage_path == storage_path_iens_1

        # Store global file
        global_resource_uri = storage.store(file_name)
        expected_uri = os.path.join(storage_path, "global", file_name)
        assert expected_uri == global_resource_uri
        assert os.path.isfile(global_resource_uri)

        # Store file for realization 1
        real_resource = storage.store(file_name, 1)
        expected_uri = os.path.join(storage_path, "1", file_name)
        assert expected_uri == real_resource
        assert os.path.isfile(real_resource)

        # Retrieve resource from global storage
        storage.retrieve(global_resource_uri, target="global_resource")
        assert os.path.isfile("global_resource")

        # Retrieve resource from realization storage
        storage.retrieve(real_resource, target="realization_resource")
        assert os.path.isfile("realization_resource")
