import os
import shutil


def storage_driver_factory(config, run_path):
    if config.get("type") == "shared_disk":
        storage_path = config["storage_path"]
        return _SharedDiskStorageDriver(storage_path, run_path)
    else:
        raise ValueError(f"Not a valid storage type. ({config.get('type')})")


class _SharedDiskStorageDriver:
    def __init__(self, storage_path, run_path):
        self._storage_path = storage_path
        self._run_path = f"{run_path}"

    def get_storage_path(self, iens):
        if iens is None:
            return f"{self._storage_path}/global"
        return f"{self._storage_path}/{iens}"

    def store(self, local_name, iens=None):
        storage_path = self.get_storage_path(iens)
        os.makedirs(storage_path, exist_ok=True)
        storage_uri = os.path.join(storage_path, local_name)
        shutil.copyfile(os.path.join(self._run_path, local_name), storage_uri)
        return storage_uri

    def retrieve(self, storage_uri, target=None):
        if storage_uri.startswith(self._storage_path):
            target = os.path.basename(storage_uri) if target is None else target
            shutil.copyfile(storage_uri, os.path.join(self._run_path, target))
            return target
        else:
            raise ValueError(f"Storage driver can't handle file: {storage_uri}")
