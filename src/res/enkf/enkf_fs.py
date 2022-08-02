#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'enkf_fs.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from typing import List

from cwrap import BaseCClass
import numpy as np
import numpy.typing as npt

from res import ResPrototype
from res.enkf.res_config import EnsembleConfig
from res.enkf.enums import EnKFFSType
from res.enkf.state_map import StateMap
from res.enkf.summary_key_set import SummaryKeySet
from res.enkf.util import TimeMap

from res._lib import update


class EnkfFs(BaseCClass):
    TYPE_NAME = "enkf_fs"

    _mount = ResPrototype("void* enkf_fs_mount(char*, bool)", bind=False)
    _sync = ResPrototype("void enkf_fs_sync(enkf_fs)")
    _decref = ResPrototype("int   enkf_fs_decref(enkf_fs)")
    _incref = ResPrototype("int   enkf_fs_incref(enkf_fs)")
    _get_refcount = ResPrototype("int   enkf_fs_get_refcount(enkf_fs)")
    _get_case_name = ResPrototype("char* enkf_fs_get_case_name(enkf_fs)")
    _is_read_only = ResPrototype("bool  enkf_fs_is_read_only(enkf_fs)")
    _is_running = ResPrototype("bool  enkf_fs_is_running(enkf_fs)")
    _fsync = ResPrototype("void  enkf_fs_fsync(enkf_fs)")
    _create = ResPrototype(
        "enkf_fs_obj   enkf_fs_create_fs(char* , enkf_fs_type_enum , bool)",
        bind=False,
    )
    _get_time_map = ResPrototype("time_map_ref  enkf_fs_get_time_map(enkf_fs)")
    _get_state_map = ResPrototype("state_map_ref enkf_fs_get_state_map(enkf_fs)")
    _summary_key_set = ResPrototype(
        "summary_key_set_ref enkf_fs_get_summary_key_set(enkf_fs)"
    )

    def __init__(self, mount_point: str, read_only: bool = False):
        c_ptr = self._mount(mount_point, read_only)
        super().__init__(c_ptr)

    def copy(self):
        fs = self.createPythonObject(self._address())
        self._incref()
        return fs

    # This method will return a new Python object which shares the underlying
    # enkf_fs instance as self. The name weakref is used because the Python
    # object returned from this method does *not* manipulate the reference
    # count of the underlying enkf_fs instance, and specifically it does not
    # inhibit destruction of this object.
    def weakref(self):
        fs = self.createCReference(self._address())
        return fs

    def getTimeMap(self) -> TimeMap:
        return self._get_time_map().setParent(self)

    def getStateMap(self) -> StateMap:
        return self._get_state_map().setParent(self)

    def getCaseName(self) -> str:
        return self._get_case_name()

    def isReadOnly(self) -> bool:
        return self._is_read_only()

    def refCount(self) -> int:
        return self._get_refcount()

    def is_running(self) -> bool:
        return self._is_running()

    @classmethod
    def createFileSystem(cls, path: str, mount: bool = False) -> "EnkfFs":
        assert isinstance(path, str)
        fs_type = EnKFFSType.BLOCK_FS_DRIVER_ID
        fs = cls._create(path, fs_type, mount)
        return fs

    def sync(self):
        self._sync()

    # The umount( ) method should not normally be called explicitly by
    # downstream code, but in situations where file descriptors is at premium
    # it might be beneficial to call it explicitly. In that case it is solely
    # the responsability of the calling scope to ensure that it is not called
    # repeatedly - that will lead to hard failure!
    def umount(self):
        if self.isReference():
            raise AssertionError(
                "Calling umount() on a reference is an application error"
            )

        if self:
            self._decref()
            self._invalidateCPointer()
        else:
            raise AssertionError("Tried to umount for second time - application error")

    def free(self):
        if self:
            self.umount()

    def __repr__(self):
        return f"EnkfFs(case_name = {self.getCaseName()}) {self._ad_str()}"

    def fsync(self):
        self._fsync()

    def getSummaryKeySet(self) -> SummaryKeySet:
        return self._summary_key_set().setParent(self)

    def realizationList(self, state):
        """
        Will return list of realizations with state == the specified state.
        @type state: res.enkf.enums.RealizationStateEnum
        @rtype: ecl.util.IntVector
        """
        state_map = self.getStateMap()
        return state_map.realizationList(state)

    def save_parameters(
        self,
        ensemble_config: EnsembleConfig,
        iens_active_index: List[int],
        parameters: List[update.Parameter],
        values: npt.ArrayLike,
    ):
        update.save_parameters(
            self, ensemble_config, iens_active_index, parameters, values
        )

    def load_parameters(
        self,
        ensemble_config: EnsembleConfig,
        iens_active_index: List[int],
        parameters: List[update.Parameter],
    ) -> np.ndarray:
        return update.load_parameters(
            self, ensemble_config, iens_active_index, parameters
        )
