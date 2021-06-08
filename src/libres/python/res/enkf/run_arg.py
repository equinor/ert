#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'run_arg.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res import ResPrototype


class RunArg(BaseCClass):
    TYPE_NAME = "run_arg"

    _alloc_ENSEMBLE_EXPERIMENT = ResPrototype(
        "run_arg_obj run_arg_alloc_ENSEMBLE_EXPERIMENT(char*, enkf_fs, int, int, char*, char*, subst_list)",
        bind=False,
    )
    _free = ResPrototype("void run_arg_free(run_arg)")
    _get_queue_index_safe = ResPrototype("int  run_arg_get_queue_index_safe(run_arg)")
    _set_queue_index = ResPrototype("void run_arg_set_queue_index(run_arg, int)")
    _is_submitted = ResPrototype("bool run_arg_is_submitted(run_arg)")
    _get_run_id = ResPrototype("char* run_arg_get_run_id(run_arg)")
    _get_geo_id = ResPrototype("int run_arg_get_geo_id(run_arg)")
    _set_geo_id = ResPrototype("void run_arg_set_geo_id(run_arg, int)")
    _get_runpath = ResPrototype("char* run_arg_get_runpath(run_arg)")
    _get_iter = ResPrototype("int run_arg_get_iter(run_arg)")
    _get_iens = ResPrototype("int run_arg_get_iens(run_arg)")
    _get_status = ResPrototype("int run_arg_get_run_status(run_arg)")
    _get_job_name = ResPrototype("char* run_arg_get_job_name(run_arg)")

    def __init__(self):
        raise NotImplementedError("Cannot instantiat RunArg directly!")

    @classmethod
    def createEnsembleExperimentRunArg(
        cls, run_id, fs, iens, runpath, jobname, subst_list, iter=0
    ):
        return cls._alloc_ENSEMBLE_EXPERIMENT(
            run_id, fs, iens, iter, runpath, jobname, subst_list
        )

    def free(self):
        self._free()

    def set_queue_index(self, index):
        self._set_queue_index(index)

    def getQueueIndex(self):
        qi = self._get_queue_index_safe()
        if qi < 0:
            raise ValueError("Cannot get queue index before job is submitted.")
        return qi

    def isSubmitted(self):
        return self._is_submitted()

    def __repr__(self):
        if self.isSubmitted():
            su = "submitted"
            qi = self.getQueueIndex()
        else:
            su = "not submitted"
            qi = "--"

        return "RunArg(queue_index = %s, %s) %s" % (qi, su, self._ad_str())

    def get_run_id(self):
        return self._get_run_id()

    @property
    def geo_id(self):
        return self._get_geo_id()

    @geo_id.setter
    def geo_id(self, value):
        self._set_geo_id(value)

    @property
    def runpath(self):
        return self._get_runpath()

    @property
    def iter_id(self):
        return self._get_iter()

    @property
    def iens(self):
        return self._get_iens()

    @property
    def run_status(self):
        return self._get_status()

    @property
    def job_name(self):
        return self._get_job_name()
