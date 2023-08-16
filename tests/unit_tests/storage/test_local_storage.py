import pytest
import xarray as xr

from ert.storage import NotifierType, StorageReader
from ert.storage import local_storage as local
from ert.storage import open_storage


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

    with open_storage(tmp_path) as storage_reader:
        with pytest.raises(TypeError):
            storage_reader.to_accessor()

    with open_storage(tmp_path, mode="w") as storage_accessor:
        storage_reader: StorageReader = storage_accessor
        storage_reader.to_accessor()


def test_notifier(tmp_path, mocker):
    notifier: NotifierType = {
        "experiment:create": mocker.Mock(),
        "ensemble:create": mocker.Mock(),
        "parameters:create": mocker.Mock(),
        "responses:create": mocker.Mock(),
    }

    def call_count():
        return [x.call_count for x in notifier.values()]

    with open_storage(tmp_path, mode="w", notifier=notifier) as storage:
        assert call_count() == [0, 0, 0, 0]

        experiment = storage.create_experiment()
        assert call_count() == [1, 0, 0, 0]

        ensemble = experiment.create_ensemble(ensemble_size=1, name="default")
        assert call_count() == [1, 1, 0, 0]

        ensemble.save_parameters("foo", 0, xr.Dataset({"values": [1]}))
        assert call_count() == [1, 1, 1, 0]

        ensemble.save_response("foo", xr.Dataset({"values": [1]}), 0)
        assert call_count() == [1, 1, 1, 1]
