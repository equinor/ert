import os.path
import re
import warnings
from pathlib import Path

from typing import List, Dict, Union, TYPE_CHECKING

from cwrap import BaseCClass
from ecl.util.util import StringList

from res import ResPrototype
from res import _lib
from res._lib import enkf_state
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import RealizationStateEnum
from res.enkf.ert_run_context import RunContext
from res.enkf.state_map import StateMap

if TYPE_CHECKING:
    from res.enkf import EnKFMain


def naturalSortKey(s: str) -> List[Union[int, str]]:
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


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


# For normal use from ert all filesystems will be located in the same
# folder in the filesystem - corresponding to the ENSPATH setting in
# the config file; in this implementation that setting is stored in
# the @mount_root field. Currently @mount_root is fixed to the value
# returned by EnKFMain.getMountPoint(), but in principle a different
# path could be sent as the the optional second argument to the
# getFS() method.


class EnkfFsManager(BaseCClass):
    TYPE_NAME = "enkf_fs_manager"

    _get_current_fs = ResPrototype("enkf_fs_obj enkf_main_get_fs_ref(enkf_fs_manager)")
    _switch_fs = ResPrototype("void enkf_main_set_fs(enkf_fs_manager, enkf_fs, char*)")

    _is_case_initialized = ResPrototype(
        "bool enkf_main_case_is_initialized(enkf_fs_manager, char*)"
    )
    _initialize_case_from_existing = ResPrototype(
        "void enkf_main_init_case_from_existing(enkf_fs_manager, enkf_fs, int, enkf_fs)"
    )
    _initialize_current_case_from_existing = ResPrototype(
        "void enkf_main_init_current_case_from_existing(enkf_fs_manager, enkf_fs, int)"
    )

    _alloc_readonly_state_map = ResPrototype(
        "state_map_obj enkf_main_alloc_readonly_state_map(enkf_fs_manager, char*)"
    )

    DEFAULT_CAPACITY = 5

    def __init__(self, enkf_main: "EnKFMain", capacity: int = DEFAULT_CAPACITY):
        # enkf_main should be an EnKFMain, get the _RealEnKFMain object
        real_enkf_main = enkf_main.parent()

        super().__init__(
            real_enkf_main.from_param(real_enkf_main).value,
            parent=real_enkf_main,
            is_reference=True,
        )

        self._fs_rotator = FileSystemRotator(capacity)
        self._mount_root = real_enkf_main.getMountPoint()

    def _createFullCaseName(self, mount_root: str, case_name: str) -> str:
        return os.path.join(mount_root, case_name)

    # The return value from the getFileSystem will be a weak reference to the
    # underlying enkf_fs object. That implies that the fs manager must be in
    # scope for the return value to be valid.
    def getFileSystem(
        self, case_name: str, mount_root: str = None, read_only: bool = False
    ) -> EnkfFs:
        if mount_root is None:
            mount_root = self._mount_root

        full_case_name = self._createFullCaseName(mount_root, case_name)

        if full_case_name not in self._fs_rotator:
            if not os.path.exists(full_case_name):
                if self._fs_rotator.atCapacity():
                    self._fs_rotator.dropOldestFileSystem()

                EnkfFs.createFileSystem(full_case_name)

            new_fs = EnkfFs(full_case_name, read_only)
            self._fs_rotator.addFileSystem(new_fs, full_case_name)

        fs = self._fs_rotator[full_case_name]

        return fs

    def isCaseRunning(self, case_name: str, mount_root: str = None) -> bool:
        """Returns true if case is mounted and write_count > 0"""
        if self.isCaseMounted(case_name, mount_root):
            case_fs = self.getFileSystem(case_name, mount_root)
            return case_fs.is_running()
        return False

    def caseExists(self, case_name: str) -> bool:
        return case_name in self.getCaseList()

    def caseHasData(self, case_name: str) -> bool:
        state_map = self.getStateMapForCase(case_name)

        return any(state == RealizationStateEnum.STATE_HAS_DATA for state in state_map)

    def getCurrentFileSystem(self) -> EnkfFs:
        """Returns the currently selected file system"""
        current_fs = self._get_current_fs()
        case_name = current_fs.getCaseName()
        full_name = self._createFullCaseName(self._mount_root, case_name)
        self.parent().addDataKW("<ERT-CASE>", current_fs.getCaseName())
        self.parent().addDataKW("<ERTCASE>", current_fs.getCaseName())

        if full_name not in self._fs_rotator:
            self._fs_rotator.addFileSystem(current_fs, full_name)

        return self.getFileSystem(case_name, self._mount_root)

    def umount(self) -> None:
        self._fs_rotator.umountAll()

    def getFileSystemCount(self) -> int:
        return len(self._fs_rotator)

    def getEnsembleSize(self) -> int:
        return self.parent().getEnsembleSize()

    def switchFileSystem(self, file_system: EnkfFs) -> None:
        self.parent().addDataKW("<ERT-CASE>", file_system.getCaseName())
        self.parent().addDataKW("<ERTCASE>", file_system.getCaseName())
        self._switch_fs(file_system, None)

    def isCaseInitialized(self, case: str) -> bool:
        return self._is_case_initialized(case)

    def getCaseList(self) -> List[str]:
        caselist = [str(x.stem) for x in Path(self._mount_root).iterdir() if x.is_dir()]
        return sorted(caselist, key=naturalSortKey)

    def customInitializeCurrentFromExistingCase(
        self,
        source_case: str,
        source_report_step: int,
        member_mask: List[bool],
        node_list: List[str],
    ) -> None:
        if source_case not in self.getCaseList():
            raise KeyError(
                f"No such source case: {source_case} in {self.getCaseList()}"
            )
        if isinstance(node_list, StringList):
            warnings.warn(
                "Using StringList for node_list is deprecated, "
                "use a python list of strings.",
                DeprecationWarning,
            )
            node_list = list(node_list)
        source_case_fs = self.getFileSystem(source_case)
        _lib.enkf_main.init_current_case_from_existing_custom(
            self, source_case_fs, source_report_step, node_list, member_mask
        )

    def initializeCurrentCaseFromExisting(
        self, source_fs: EnkfFs, source_report_step: int
    ) -> None:
        self._initialize_current_case_from_existing(source_fs, source_report_step)

    def initializeCaseFromExisting(
        self, source_fs: EnkfFs, source_report_step: int, target_fs: EnkfFs
    ) -> None:
        self._initialize_case_from_existing(source_fs, source_report_step, target_fs)

    def initializeFromScratch(
        self, parameter_list: List[str], run_context: RunContext
    ) -> None:
        if isinstance(parameter_list, StringList):
            warnings.warn(
                "Using StringList for node_list is deprecated, "
                "use a python list of strings.",
                DeprecationWarning,
            )
            parameter_list = list(parameter_list)
        for realization_nr in range(self.parent().getEnsembleSize()):
            if run_context.is_active(realization_nr):
                enkf_state.state_initialize(
                    self.parent(),
                    run_context.sim_fs,
                    parameter_list,
                    run_context.init_mode.value,
                    realization_nr,
                )

    def isCaseMounted(self, case_name: str, mount_root: str = None) -> bool:
        if mount_root is None:
            mount_root = self._mount_root

        full_case_name = self._createFullCaseName(mount_root, case_name)

        return full_case_name in self._fs_rotator

    def getStateMapForCase(self, case: str) -> StateMap:
        if self.isCaseMounted(case):
            fs = self.getFileSystem(case)
            return fs.getStateMap()
        else:
            return self._alloc_readonly_state_map(case)

    def isCaseHidden(self, case_name: str) -> bool:
        return case_name.startswith(".")
