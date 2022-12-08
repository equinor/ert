import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, List, Optional

from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._clib.state_map import RealizationStateEnum

if TYPE_CHECKING:
    from ecl.summary import EclSum

    from ert._c_wrappers.enkf import EnsembleConfig
    from ert._clib.state_map import StateMap

FS_VERSION = 0
FS_VERSION_FILE = ".fs_version"

logger = logging.getLogger(__file__)


class FileSystemError(Exception):
    pass


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
        refcase: Optional["EclSum"] = None,
    ):
        self.capacity = capacity
        self.storage_path = storage_path
        self.refcase = refcase
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)
        self._check_version()
        self.read_only = read_only
        self._ensemble_config = ensemble_config
        self._ensemble_size = ensemble_size
        current_case_file = storage_path / "current_case"
        if current_case_file.exists():
            mount_path = storage_path / current_case_file.read_text("utf-8").strip()
        else:
            mount_path = storage_path / "default"
        if mount_path.exists():
            fs = EnkfFs(
                mount_path,
                self._ensemble_config,
                self._ensemble_size,
                read_only=read_only,
                refcase=self.refcase,
            )
        else:
            fs = EnkfFs.createFileSystem(
                mount_path,
                self._ensemble_config,
                self._ensemble_size,
                read_only=read_only,
                refcase=self.refcase,
            )

        self.open_storages: Dict[str, EnkfFs] = {fs.case_name: fs}
        self.active_case = fs.case_name

    @property
    def current_case(self) -> "EnkfFs":
        return self[self.active_case]

    @property
    def cases(self) -> List[str]:
        return [x.stem for x in Path(self.storage_path).iterdir() if x.is_dir()]

    def _check_version(self) -> None:
        """
        Checks if the file system on disk was created with the same fs version
        currently in use.
        """
        version_file = self.storage_path / FS_VERSION_FILE
        if not version_file.exists():
            # If the version file does not exist it uses old fs
            with open(version_file, "w", encoding="utf-8") as f:
                json.dump({"version": 0}, f)
        with open(version_file, "r", encoding="utf-8") as f:
            version_data = json.load(f)
        if version_data["version"] < FS_VERSION:
            raise FileSystemError(
                "Trying to load storage created by an older"
                f" file system version: Current version: {FS_VERSION}"
                f", found on disk: {version_data['version']}"
            )
        if version_data["version"] > FS_VERSION:
            raise FileSystemError(
                "Trying to load storage created by a newer"
                f" file system version: Current version: {FS_VERSION}"
                f", found on disk: {version_data['version']}"
            )

    def __len__(self) -> int:
        return len(self.cases)

    def add_case(self, case_name: str) -> "EnkfFs":
        if case_name in self:
            raise ValueError(f"Duplicate case: {case_name} in {self.cases}")
        file_system = EnkfFs.createFileSystem(
            self.storage_path / case_name,
            self._ensemble_config,
            self._ensemble_size,
            self.read_only,
            self.refcase,
        )

        self._add_to_open(file_system)
        return file_system

    def _add_to_open(self, file_system: "EnkfFs") -> None:
        if len(self.open_storages) == self.capacity:
            self._drop_oldest_file_system()
        self.open_storages[file_system.case_name] = file_system

    def state_map(self, case_name: str) -> "StateMap":
        return self[case_name].getStateMap()

    def has_data(self, case: str) -> bool:
        state_map = self.state_map(case)
        return RealizationStateEnum.STATE_HAS_DATA in state_map

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
            file_system = EnkfFs(
                self.storage_path / case_name,
                self._ensemble_config,
                self._ensemble_size,
                self.read_only,
                self.refcase,
            )
            self._add_to_open(file_system)
            return file_system
        else:
            raise KeyError(f"No such case name: {case_name} in {self.cases}")

    def __iter__(self) -> Generator:
        yield from self.cases
