from typing import List, Dict, Union

from ert._c_wrappers.enkf.enkf_fs import EnkfFs


class FileSystemRotator:
    def __init__(self, capacity: int):
        super().__init__()
        self._capacity: int = capacity
        self._fs_list: List[str] = []
        self._fs_map: Dict[str, EnkfFs] = {}

    def __len__(self) -> int:
        return len(self._fs_list)

    def addFileSystem(self, file_system: EnkfFs, full_name: str) -> None:
        if self.atCapacity():
            self.dropOldestFileSystem()

        self._fs_list.append(full_name)
        self._fs_map[full_name] = file_system

    def dropOldestFileSystem(self) -> None:
        if len(self._fs_list) > 0:
            case_name = self._fs_list[0]
            del self._fs_list[0]
            self._fs_map[case_name].sync()
            del self._fs_map[case_name]

    def atCapacity(self) -> bool:
        return len(self._fs_list) == self._capacity

    def __contains__(self, full_case_name: str) -> bool:
        return full_case_name in self._fs_list

    def __get_fs(self, name: str) -> EnkfFs:
        fs = self._fs_map[name]
        return fs.copy()

    def __getitem__(self, case: Union[int, str]) -> EnkfFs:
        if isinstance(case, str):
            return self.__get_fs(case)
        elif isinstance(case, int) and 0 <= case < len(self):
            case_name = self._fs_list[case]
            return self.__get_fs(case_name)
        else:
            raise IndexError(f"Value '{case}' is not a proper index or case name.")

    def umountAll(self) -> None:
        while len(self._fs_list) > 0:
            self.dropOldestFileSystem()
