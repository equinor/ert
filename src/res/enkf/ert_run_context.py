#  Copyright (C) 2014  Equinor ASA, Norway.
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
from typing import List, Optional

from cwrap import BaseCClass
from res import ResPrototype, _lib
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import EnkfInitModeEnum, EnkfRunType
from res.enkf.run_arg import RunArg


class ErtRunContext(BaseCClass):
    TYPE_NAME = "ert_run_context"
    _alloc = ResPrototype(
        "void* ert_run_context_alloc_empty(enkf_run_mode_enum , \
                                     enkf_init_mode_enum, \
                                     int)",
        bind=False,
    )
    _get_size = ResPrototype("int ert_run_context_get_size( ert_run_context )")
    _free = ResPrototype("void ert_run_context_free( ert_run_context )")
    _iactive = ResPrototype("bool ert_run_context_iactive( ert_run_context , int)")
    _iget = ResPrototype("run_arg_ref ert_run_context_iget_arg( ert_run_context , int)")
    _get_id = ResPrototype("char* ert_run_context_get_id( ert_run_context )")
    _get_iter = ResPrototype("int ert_run_context_get_iter( ert_run_context )")
    _get_target_fs = ResPrototype(
        "enkf_fs_ref ert_run_context_get_update_target_fs( ert_run_context )"
    )
    _get_sim_fs = ResPrototype(
        "enkf_fs_ref ert_run_context_get_sim_fs( ert_run_context )"
    )
    _get_init_mode = ResPrototype(
        "enkf_init_mode_enum ert_run_context_get_init_mode( ert_run_context )"
    )

    _get_step = ResPrototype("int ert_run_context_get_step1(ert_run_context)")
    _deactivate_realization = ResPrototype(
        "void ert_run_context_deactivate_realization( ert_run_context, int)"
    )

    def __init__(
        self,
        run_type: EnkfRunType,
        sim_fs: Optional[EnkfFs],
        target_fs: Optional[EnkfFs],
        mask: List[bool],
        paths: List[str],
        jobnames: List[str],
        itr: int = 0,
        init_mode=EnkfInitModeEnum.INIT_CONDITIONAL,
    ):
        c_ptr = self._alloc(
            run_type,
            init_mode,
            itr,
        )
        super().__init__(c_ptr)

        _lib.ert_run_context.set_active(self, mask)
        if sim_fs is not None:
            _lib.ert_run_context.set_sim_fs(self, sim_fs)
        if target_fs is not None:
            _lib.ert_run_context.set_target_fs(self, target_fs)

        if run_type == EnkfRunType.ENSEMBLE_EXPERIMENT:
            _lib.ert_run_context.add_ensemble_experiment_args(self, paths, jobnames)
        elif run_type == EnkfRunType.SMOOTHER_RUN:
            _lib.ert_run_context.add_smoother_run_args(
                self,
                paths,
                jobnames,
            )
        elif run_type == EnkfRunType.INIT_ONLY:
            _lib.ert_run_context.add_init_only_args(self, paths)
        elif run_type == EnkfRunType.SMOOTHER_UPDATE:
            pass
        elif run_type == EnkfRunType.CASE_INIT_ONLY:
            pass
        else:
            raise ValueError(f"Unsupported run type {run_type}")

    @classmethod
    def ensemble_experiment(
        cls, sim_fs, mask: List[bool], paths, jobnames, itr
    ) -> "ErtRunContext":
        return cls(
            run_type=EnkfRunType.ENSEMBLE_EXPERIMENT,
            sim_fs=sim_fs,
            target_fs=None,
            mask=mask,
            paths=paths,
            jobnames=jobnames,
            itr=itr,
        )

    @classmethod
    def ensemble_smoother(
        cls, sim_fs, target_fs, mask: List[bool], paths, jobnames, itr
    ) -> "ErtRunContext":
        return cls(
            EnkfRunType.SMOOTHER_RUN,
            sim_fs,
            target_fs,
            mask,
            paths,
            jobnames,
            itr,
        )

    @classmethod
    def ensemble_smoother_update(
        cls,
        sim_fs,
        target_fs,
    ):
        return cls(
            run_type=EnkfRunType.SMOOTHER_UPDATE,
            mask=[],
            sim_fs=sim_fs,
            target_fs=target_fs,
            paths=[],
            jobnames=[],
        )

    @classmethod
    def case_init(cls, sim_fs, mask=None):
        if mask == None:
            mask = []
        return cls(
            run_type=EnkfRunType.CASE_INIT_ONLY,
            init_mode=EnkfInitModeEnum.INIT_FORCE,
            mask=mask,
            sim_fs=sim_fs,
            target_fs=None,
            paths=[],
            jobnames=[],
        )

    def is_active(self, index: int) -> bool:
        return self._iactive(index)

    def __len__(self):
        return self._get_size()

    def __getitem__(self, index) -> RunArg:
        if not isinstance(index, int):
            raise TypeError("Invalid type - expected integer")

        if 0 <= index < len(self):
            run_arg = self._iget(index)
            return run_arg
        else:
            raise IndexError(f"Index:{index} invalid. Legal range: [0,{len(self)})")

    def free(self):
        self._free()

    def __repr__(self):
        return f"ErtRunContext(size = {len(self)}) {self._ad_str()}"

    def get_id(self):
        return self._get_id()

    def get_mask(self) -> List[bool]:
        return [self.is_active(i) for i in range(len(self))]

    def get_iter(self) -> int:
        return self._get_iter()

    def get_target_fs(self) -> EnkfFs:
        return self._get_target_fs()

    def get_sim_fs(self) -> EnkfFs:
        return self._get_sim_fs()

    def get_init_mode(self):
        return self._get_init_mode()

    def get_step(self):
        return self._get_step()

    def deactivate_realization(self, realization_nr):
        self._deactivate_realization(realization_nr)
