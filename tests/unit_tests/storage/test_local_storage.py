import pytest

from ert.storage import StorageReader, open_storage
from ert.storage import local_storage as local


def _cases(storage):
    return sorted(x.name for x in storage.ensembles)


def test_open_empty_reader(tmp_path):
    with open_storage(tmp_path / "empty", mode="r") as storage:
        assert _cases(storage) == []

    # StorageReader doesn't create an empty directory
    assert not (tmp_path / "empty").is_dir()


def test_open_empty_accessor(tmp_path):
    with open_storage(tmp_path / "empty", mode="w") as storage:
        assert _cases(storage) == []

    # StorageAccessor creates the directory
    assert (tmp_path / "empty").is_dir()


def test_create_ensemble(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        assert _cases(storage) == []
        experiment_id = storage.create_experiment()
        storage.create_ensemble(experiment_id, name="foo", ensemble_size=42)
        assert _cases(storage) == ["foo"]

    with open_storage(tmp_path) as storage:
        assert _cases(storage) == ["foo"]


def test_lock(tmp_path, monkeypatch):
    with open_storage(tmp_path, mode="w") as storage:
        experiment_id = storage.create_experiment()
        storage.create_ensemble(experiment_id, name="foo", ensemble_size=42)

        # Opening with write access will timeout when opening lock
        monkeypatch.setattr(local.LocalStorageAccessor, "LOCK_TIMEOUT", 0.1)
        with pytest.raises(TimeoutError):
            open_storage(tmp_path, mode="w")

        # Opening with read-only access is fine
        with open_storage(tmp_path) as storage2:
            assert _cases(storage) == _cases(storage2)

    # Opening storage after the other instance is closed is fine
    with open_storage(tmp_path, mode="w") as storage:
        assert _cases(storage) == ["foo"]


def test_refresh(tmp_path):
    with open_storage(tmp_path, mode="w") as accessor:
        experiment_id = accessor.create_experiment()
        with open_storage(tmp_path, mode="r") as reader:
            assert _cases(accessor) == _cases(reader)

            accessor.create_ensemble(experiment_id, name="foo", ensemble_size=42)
            # Reader does not know about the newly created ensemble
            assert _cases(accessor) != _cases(reader)

            reader.refresh()
            # Reader knows about it after the refresh
            assert _cases(accessor) == _cases(reader)


def test_runtime_types(tmp_path):
    with open_storage(tmp_path) as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert not isinstance(storage, local.LocalStorageAccessor)

    with open_storage(tmp_path, mode="r") as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert not isinstance(storage, local.LocalStorageAccessor)

    with open_storage(tmp_path, mode="w") as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert isinstance(storage, local.LocalStorageAccessor)


def test_to_accessor(tmp_path):
    """
    Type-correct casting from StorageReader to StorageAccessor in cases where a
    function accepts StorageReader, but has additional functionality if it's a
    StorageAccessor. Eg, in the ERT GUI, we may pass StorageReader to the
    CaseList widget, which lists which ensembles are available, but if
    .to_accessor() doesn't throw then CaseList can also create new ensembles.
    """

    with open_storage(tmp_path) as storage_reader, pytest.raises(TypeError):
        storage_reader.to_accessor()

    with open_storage(tmp_path, mode="w") as storage_accessor:
        storage_reader: StorageReader = storage_accessor
        storage_reader.to_accessor()
