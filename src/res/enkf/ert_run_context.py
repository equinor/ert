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
from cwrap import BaseCClass

from ecl.util.util import StringList

from res.util import PathFormat
from res import ResPrototype
from res.enkf import TimeMap, StateMap, RunArg
from res.enkf.enums import EnkfInitModeEnum


class ErtRunContext(BaseCClass):
    TYPE_NAME = "ert_run_context"
    _alloc = ResPrototype(
        "void* ert_run_context_alloc( enkf_run_mode_enum , enkf_init_mode_enum, enkf_fs , enkf_fs, bool_vector, path_fmt ,char*, subst_list, int)",
        bind=False,
    )
    _alloc_ensemble_experiment = ResPrototype(
        "ert_run_context_obj ert_run_context_alloc_ENSEMBLE_EXPERIMENT( enkf_fs, bool_vector, path_fmt ,char*, subst_list, int)",
        bind=False,
    )
    _alloc_ensemble_smoother = ResPrototype(
        "ert_run_context_obj ert_run_context_alloc_SMOOTHER_RUN( enkf_fs , enkf_fs, bool_vector, path_fmt ,char*, subst_list, int)",
        bind=False,
    )
    _alloc_ensemble_smoother_update = ResPrototype(
        "ert_run_context_obj ert_run_context_alloc_SMOOTHER_UPDATE(enkf_fs , enkf_fs )",
        bind=False,
    )
    _alloc_case_init = ResPrototype(
        "ert_run_context_obj ert_run_context_alloc_CASE_INIT(enkf_fs, bool_vector)",
        bind=False,
    )
    _alloc_runpath_list = ResPrototype(
        "stringlist_obj ert_run_context_alloc_runpath_list(bool_vector, path_fmt, subst_list, int)",
        bind=False,
    )
    _alloc_runpath = ResPrototype(
        "char* ert_run_context_alloc_runpath(int, path_fmt, subst_list, int)",
        bind=False,
    )
    _get_size = ResPrototype("int ert_run_context_get_size( ert_run_context )")
    _free = ResPrototype("void ert_run_context_free( ert_run_context )")
    _iactive = ResPrototype("bool ert_run_context_iactive( ert_run_context , int)")
    _iget = ResPrototype("run_arg_ref ert_run_context_iget_arg( ert_run_context , int)")
    _get_id = ResPrototype("char* ert_run_context_get_id( ert_run_context )")
    _get_mask = ResPrototype(
        "bool_vector_obj ert_run_context_alloc_iactive( ert_run_context )"
    )
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
        run_type,
        sim_fs,
        target_fs,
        mask,
        path_fmt,
        jobname_fmt,
        subst_list,
        itr,
        init_mode=EnkfInitModeEnum.INIT_CONDITIONAL,
    ):
        c_ptr = self._alloc(
            run_type,
            init_mode,
            sim_fs,
            target_fs,
            mask,
            path_fmt,
            jobname_fmt,
            subst_list,
            itr,
        )
        super(ErtRunContext, self).__init__(c_ptr)

        # The C object ert_run_context uses a shared object for the
        # path_fmt and subst_list objects. We therefor hold on
        # to a reference here - to inhibt Python GC of these objects.
        self._path_fmt = path_fmt
        self._subst_list = subst_list

    @classmethod
    def case_init(cls, sim_fs, mask):
        return cls._alloc_case_init(sim_fs, mask)

    @classmethod
    def ensemble_experiment(cls, sim_fs, mask, path_fmt, jobname_fmt, subst_list, itr):
        run_context = cls._alloc_ensemble_experiment(
            sim_fs, mask, path_fmt, jobname_fmt, subst_list, itr
        )

        # The C object ert_run_context uses a shared object for the
        # path_fmt and subst_list objects. We therefor hold on
        # to a reference here - to inhibt Python GC of these objects.
        run_context._path_fmt = path_fmt
        run_context._subst_list = subst_list

        return run_context

    @classmethod
    def ensemble_smoother(
        cls, sim_fs, target_fs, mask, path_fmt, jobname_fmt, subst_list, itr
    ):
        run_context = cls._alloc_ensemble_smoother(
            sim_fs, target_fs, mask, path_fmt, jobname_fmt, subst_list, itr
        )

        # The C object ert_run_context uses a shared object for the
        # path_fmt and subst_list objects. We therefor hold on
        # to a reference here - to inhibt Python GC of these objects.
        run_context._path_fmt = path_fmt
        run_context._subst_list = subst_list

        return run_context

    @classmethod
    def ensemble_smoother_update(cls, sim_fs, target_fs):
        return cls._alloc_ensemble_smoother_update(sim_fs, target_fs)

    def is_active(self, index):
        if 0 <= index < len(self):
            return self._iactive(index)
        else:
            raise IndexError(
                "Index:%d invalid. Legal range: [0,%d)" % (index, len(self))
            )

    def __len__(self):
        return self._get_size()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Invalid type - expetected integer")

        if 0 <= index < len(self):
            run_arg = self._iget(index)
            return run_arg
        else:
            raise IndexError(
                "Index:%d invalid. Legal range: [0,%d)" % (index, len(self))
            )

    def free(self):
        self._free()

    def __repr__(self):
        return "ErtRunContext(size = %d) %s" % (len(self), self._ad_str())

    @classmethod
    def createRunpathList(cls, mask, runpath_fmt, subst_list, iter=0):
        """@rtype: ecl.util.stringlist.StringList"""
        return cls._alloc_runpath_list(mask, runpath_fmt, subst_list, iter)

    @classmethod
    def createRunpath(cls, iens, runpath_fmt, subst_list, iter=0):
        """@rtype: str"""
        return cls._alloc_runpath(iens, runpath_fmt, subst_list, iter)

    def get_id(self):
        return self._get_id()

    def get_mask(self):
        return self._get_mask()

    def get_iter(self):
        return self._get_iter()

    def get_target_fs(self):
        return self._get_target_fs()

    def get_sim_fs(self):
        return self._get_sim_fs()

    def get_init_mode(self):
        return self._get_init_mode()

    def get_step(self):
        return self._get_step()

    def deactivate_realization(self, realization_nr):
        self._deactivate_realization(realization_nr)
