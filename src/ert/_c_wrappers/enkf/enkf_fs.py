import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnKFFSType, RealizationStateEnum
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._c_wrappers.enkf.summary_key_set import SummaryKeySet
from ert._c_wrappers.enkf.time_map import TimeMap
from ert._clib import update
from ert.ensemble_evaluator.callbacks import _forward_model_ok

if TYPE_CHECKING:
    from ecl.util.util import IntVector

    from ert._c_wrappers.enkf.res_config import EnsembleConfig, ModelConfig
    from ert._c_wrappers.enkf.run_arg import RunArg
    from ert._c_wrappers.enkf.state_map import StateMap

logger = logging.getLogger(__name__)


def _load_realization(
    sim_fs: "EnkfFs",
    realisation: int,
    ensemble_config: "EnsembleConfig",
    model_config: "ModelConfig",
    run_args: List["RunArg"],
) -> Tuple["LoadStatus", str]:
    state_map = sim_fs.getStateMap()

    state_map.update_matching(
        realisation,
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_INITIALIZED,
    )
    status = _forward_model_ok(run_args[realisation], ensemble_config, model_config)
    state_map._set(
        realisation,
        RealizationStateEnum.STATE_HAS_DATA
        if status[0] == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE,
    )
    return status


class EnkfFs(BaseCClass):
    TYPE_NAME = "enkf_fs"

    _mount = ResPrototype("void* enkf_fs_mount(char*, int, bool)", bind=False)
    _sync = ResPrototype("void enkf_fs_sync(enkf_fs)")
    _fsync = ResPrototype("void  enkf_fs_fsync(enkf_fs)")
    _create = ResPrototype(
        "void*   enkf_fs_create_fs(char* , enkf_fs_type_enum ,int, bool)",
        bind=False,
    )
    _get_time_map = ResPrototype("time_map_ref  enkf_fs_get_time_map(enkf_fs)")
    _summary_key_set = ResPrototype(
        "summary_key_set_ref enkf_fs_get_summary_key_set(enkf_fs)"
    )
    _umount = ResPrototype("void enkf_fs_umount(enkf_fs)")

    def __init__(
        self,
        mount_point: Union[str, Path],
        ensemble_config: "EnsembleConfig",
        ensemble_size: int,
        read_only: bool = False,
    ):
        self.mount_point = Path(mount_point).absolute()
        self.case_name = self.mount_point.stem
        c_ptr = self._mount(self.mount_point.as_posix(), ensemble_size, read_only)
        super().__init__(c_ptr)
        self._ensemble_config = ensemble_config
        self._ensemble_size = ensemble_size

    def getTimeMap(self) -> TimeMap:
        return self._get_time_map().setParent(self)

    def getStateMap(self) -> "StateMap":
        return _clib.enkf_fs.get_state_map(self)

    def getCaseName(self) -> str:
        return self.case_name

    @property
    def is_initalized(self) -> bool:
        return _clib.enkf_fs.is_initialized(
            self,
            self._ensemble_config,
            self._ensemble_config.parameters,
            self._ensemble_size,
        )

    @classmethod
    def createFileSystem(
        cls,
        path: Union[str, Path],
        ensemble_config: "EnsembleConfig",
        ensemble_size: int,
        read_only: bool = False,
    ) -> "EnkfFs":
        path = Path(path).absolute()
        fs_type = EnKFFSType.BLOCK_FS_DRIVER_ID
        cls._create(path.as_posix(), fs_type, ensemble_size, False)
        return cls(path, ensemble_config, ensemble_size, read_only=read_only)

    def sync(self):
        self._sync()

    def free(self):
        self._umount()

    def __repr__(self):
        return f"EnkfFs(case_name = {self.getCaseName()}) {self._ad_str()}"

    def fsync(self):
        self._fsync()

    def getSummaryKeySet(self) -> SummaryKeySet:
        return self._summary_key_set().setParent(self)

    def realizationList(self, state: "RealizationStateEnum") -> "IntVector":
        """
        Will return list of realizations with state == the specified state.
        """
        state_map = self.getStateMap()
        return state_map.realizationList(state)

    def save_parameters(
        self,
        ensemble_config: "EnsembleConfig",
        iens_active_index: List[int],
        parameter: update.Parameter,
        values: npt.ArrayLike,
    ) -> None:
        update.save_parameter(
            self, ensemble_config, iens_active_index, parameter, values
        )

    def load_parameter(
        self,
        ensemble_config: "EnsembleConfig",
        iens_active_index: List[int],
        parameter: update.Parameter,
    ) -> np.ndarray:
        return update.load_parameter(
            self, ensemble_config, iens_active_index, parameter
        )

    def copy_from_case(
        self, other: "EnkfFs", report_step: int, nodes: List[str], active: List[bool]
    ):
        """
        This copies parameters from self into other, checking if nodes exists
        in self before performing copy.
        """
        _clib.enkf_fs.copy_from_case(
            self,
            self._ensemble_config,
            other,
            report_step,
            nodes,
            active,
        )

    def load_from_run_path(
        self,
        ensemble_size: int,
        ensemble_config: "EnsembleConfig",
        model_config: "ModelConfig",
        run_args: List["RunArg"],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""

        loaded = 0
        for iens in range(ensemble_size):
            if not active_realizations[iens]:
                continue
            result = _load_realization(
                self, iens, ensemble_config, model_config, run_args
            )
            if result[0] == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
            else:
                logger.error(f"Realization: {iens}, load failure: {result[1]}")

        return loaded
