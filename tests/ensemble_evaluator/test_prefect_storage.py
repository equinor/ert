from pathlib import Path
import pytest
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)


def test_storage_driver(tmpdir):
    storage_path = Path(".my_storage_test_path")
    test_config = {"type": "shared_disk", "storage_path": storage_path}

    with tmpdir.as_cwd():
        file_name = "test_file.txt"
        with open(file_name, "w") as f:
            f.write("...")

        storage = storage_driver_factory(config=test_config, run_path=tmpdir.strpath)

        assert storage is not None

        # Check storage path
        global_storage_path = storage.get_storage_path(None)
        expected_global_storage_path = storage_path / "global"
        assert expected_global_storage_path == global_storage_path

        storage_path_iens_1 = storage.get_storage_path(1)
        expected_storage_path = storage_path / "1"
        assert expected_storage_path == storage_path_iens_1

        # Store global file
        global_resource_uri = storage.store(file_name)
        expected_uri = storage_path / "global" / file_name
        assert expected_uri == global_resource_uri
        assert global_resource_uri.is_file()

        # Store file for realization 1
        real_resource = storage.store(file_name, 1)
        expected_uri = storage_path / "1" / file_name
        assert expected_uri == real_resource
        assert real_resource.is_file()

        # Retrieve resource from global storage
        storage.retrieve(global_resource_uri, target="global_resource")
        assert Path("global_resource").is_file()

        # Retrieve resource from realization storage
        storage.retrieve(real_resource, target="realization_resource")
        assert Path("realization_resource").is_file()

        # Retrieve non-existing file
        file_uri = global_storage_path / "foo"
        assert not file_uri.is_file()
        with pytest.raises(
            ValueError, match=f"Storage does not contain file: {file_uri}"
        ):
            storage.retrieve(file_uri)

        # Retrieve a file that is not in storage, but exists outside of it.
        file_uri = Path("foo.txt")
        with open(file_uri, "w") as f:
            f.write("bar")
        assert file_uri.is_file()
        with pytest.raises(
            ValueError, match=f"Storage does not contain file: {file_uri}"
        ):
            storage.retrieve(file_uri)

        # Retrieve an existing file that is not in storage using a relative uri.
        file_uri = storage_path / ".." / "foo.txt"
        with open(file_uri, "w") as f:
            f.write("bar")
        assert file_uri.is_file()
        with pytest.raises(
            ValueError, match=f"Storage does not contain file: {file_uri}"
        ):
            storage.retrieve(file_uri)

        # Store data global data
        data = {"some_data": 42}
        data_file = "data_file"
        global_resource_uri = storage.store_data(data, data_file)
        expected_uri = storage_path / "global" / data_file
        assert expected_uri == global_resource_uri
        assert global_resource_uri.is_file()

        # Store data realization data
        data = {"some_data": 42}
        data_file = "data_file"
        real_resource = storage.store_data(data, data_file, 1)
        expected_uri = storage_path / "1" / data_file
        assert expected_uri == real_resource
        assert real_resource.is_file()
