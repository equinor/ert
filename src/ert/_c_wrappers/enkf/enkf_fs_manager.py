from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

from ert._c_wrappers.enkf.enkf_fs import EnkfFs


@dataclass
class FileSystemRotator:
    capacity: int
    fs_map: Dict[Path, EnkfFs] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.fs_map)

    def append(self, file_system: "EnkfFs") -> None:
        if len(self) == self.capacity:
            self.drop_oldest_file_system()
        self.fs_map[file_system.mount_point] = file_system

    def drop_oldest_file_system(self) -> None:
        if len(self.fs_map) > 0:
            case_name = list(self.fs_map)[0]
            self.fs_map[case_name].sync()
            del self.fs_map[case_name]

    def __contains__(self, full_case_name: Union[str, Path]) -> bool:
        return Path(full_case_name).absolute() in self.fs_map

    def __getitem__(self, case: Union[str, Path]) -> EnkfFs:
        return self.fs_map[Path(case).absolute()]

    def umount(self) -> None:
        while len(self.fs_map) > 0:
            self.drop_oldest_file_system()
