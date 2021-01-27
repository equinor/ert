import shutil
from pathlib import Path


def storage_driver_factory(config, run_path):
    if config.get("type") == "shared_disk":
        storage_path = config["storage_path"]
        return _SharedDiskStorageDriver(storage_path, run_path)
    else:
        raise ValueError(f"Not a valid storage type. ({config.get('type')})")


class _SharedDiskStorageDriver:
    def __init__(self, storage_path, run_path):
        self._storage_path = Path(storage_path)
        self._run_path = Path(run_path)

    def get_storage_path(self, iens):
        if iens is None:
            return self._storage_path / "global"
        return self._storage_path / str(iens)

    def store(self, local_name, iens=None):
        storage_path = self.get_storage_path(iens)
        storage_path.mkdir(parents=True, exist_ok=True)
        storage_uri = storage_path / local_name
        shutil.copyfile(self._run_path / local_name, storage_uri)
        return storage_uri

    def retrieve(self, storage_uri, target=None):
        storage_uri = Path(storage_uri)
        if not (
            storage_uri.is_file() and _is_relative_to(storage_uri, self._storage_path)
        ):
            raise ValueError(f"Storage does not contain file: {storage_uri}")
        target = storage_uri.name if target is None else target
        shutil.copyfile(storage_uri, self._run_path / target)
        return target


def _is_relative_to(child, parent):
    """Emulate path.is_relative_to() from Python 3.9"""
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
