from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from ert._c_wrappers.enkf.enkf_fs import EnkfFs

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig
    from ert._clib.state_map import StateMap


class FileSystemManager:
    """
    This keeps track of available storages, it has a cache of n open storages,
    and keeps track of all the available storages on disk. If the user requests
    a storage that is on disk but not in the cache, the manager will append the
    one from disk. If appending occurs and there are too many open storages, the
    oldest will be popped.
    """

    def __init__(
        self,
        capacity: int,
        storage_path: Path,
        ensemble_config: "EnsembleConfig",
        ensemble_size: int,
        read_only: bool,
    ):
        self.capacity = capacity
        self.storage_path = storage_path
        self.read_only = read_only
        self._ensemble_config = ensemble_config
        self._ensemble_size = ensemble_size
        current_case_file = storage_path / "current_case"
        if current_case_file.exists():
            fs = EnkfFs(
                storage_path / current_case_file.read_text("utf-8").strip(),
                self._ensemble_config,
                self._ensemble_size,
                read_only=read_only,
            )
        else:
            fs = EnkfFs.createFileSystem(
                storage_path / "default",
                self._ensemble_config,
                self._ensemble_size,
                read_only=read_only,
            )
        self.open_storages: Dict[str, EnkfFs] = {fs.case_name: fs}
        self.active_case = fs.case_name

    @property
    def current_case(self) -> "EnkfFs":
        return self[self.active_case]

    @property
    def cases(self) -> List[str]:
        return [x.stem for x in Path(self.storage_path).iterdir() if x.is_dir()]

    def __len__(self) -> int:
        return len(self.cases)

    def add_case(self, case_name: str) -> "EnkfFs":
        if case_name in self.open_storages:
            raise ValueError(f"Duplicate case: {case_name} in {self.open_storages}")
        if case_name in self.cases:
            file_system = EnkfFs(
                self.storage_path / case_name,
                self._ensemble_config,
                self._ensemble_size,
                self.read_only,
            )
        else:
            file_system = EnkfFs.createFileSystem(
                self.storage_path / case_name,
                self._ensemble_config,
                self._ensemble_size,
                self.read_only,
            )
        if len(self.open_storages) == self.capacity:
            self._drop_oldest_file_system()
        self.open_storages[file_system.case_name] = file_system
        return file_system

    def state_map(self, case_name: str) -> "StateMap":
        if case_name in self.open_storages:
            return self.open_storages[case_name].getStateMap()
        elif case_name in self.cases:
            return EnkfFs.read_state_map((self.storage_path / case_name).as_posix())
        else:
            raise KeyError(f"No such case: {case_name} in: {self.cases}")

    def _drop_oldest_file_system(self) -> None:
        if len(self.open_storages) > 0:
            case_name = list(self.open_storages)[0]
            self.open_storages[case_name].sync()
            del self.open_storages[case_name]

    def __contains__(self, case_name: str) -> bool:
        return case_name in self.cases

    def __getitem__(self, case_name: str) -> EnkfFs:
        if case_name in self.open_storages:
            return self.open_storages[case_name]
        elif case_name in self.cases:
            return self.add_case(case_name)
        else:
            raise KeyError(f"No such case name: {case_name} in {self.cases}")

    def umount(self) -> None:
        while len(self.open_storages) > 0:
            self._drop_oldest_file_system()
