#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'forward_model.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.job_queue import ExtJob, ExtJoblist
from res.job_queue import EnvironmentVarlist
from ecl.util.util import StringList
from res.util.substitution_list import SubstitutionList


class ForwardModel(BaseCClass):
    TYPE_NAME = "forward_model"

    _alloc = ResPrototype("void* forward_model_alloc(ext_joblist)", bind=False)
    _free = ResPrototype("void forward_model_free( forward_model )")
    _clear = ResPrototype("void forward_model_clear(forward_model)")
    _add_job = ResPrototype("ext_job_ref forward_model_add_job(forward_model, char*)")
    _alloc_joblist = ResPrototype(
        "stringlist_obj forward_model_alloc_joblist(forward_model)"
    )
    _iget_job = ResPrototype("ext_job_ref forward_model_iget_job( forward_model, int)")
    _get_length = ResPrototype("int forward_model_get_length(forward_model)")
    _formatted_fprintf = ResPrototype(
        "void forward_model_formatted_fprintf(forward_model, char*, char*, char*, subst_list, int, env_varlist)"
    )

    def __init__(self, ext_joblist):
        c_ptr = self._alloc(ext_joblist)
        if c_ptr:
            super(ForwardModel, self).__init__(c_ptr)
        else:
            raise ValueError(
                "Failed to construct forward model from provided ext_joblist %s"
                % ext_joblist
            )

    def __len__(self):
        return self._get_length()

    def joblist(self):
        """@rtype: StringList"""
        return self._alloc_joblist()

    def iget_job(self, index):
        """@rtype: ExtJob"""
        return self._iget_job(index).setParent(self)

    def add_job(self, name):
        """@rtype: ExtJob"""
        return self._add_job(name).setParent(self)

    def clear(self):
        self._clear()

    def free(self):
        self._free()

    def formatted_fprintf(
        self, run_id, path, data_root, global_args, umask, env_varlist
    ):
        self._formatted_fprintf(
            run_id, path, data_root, global_args, umask, env_varlist
        )

    def __repr__(self):
        return self._create_repr("joblist=%s" % self.joblist())

    def get_size(self):
        return len(self)

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        for i in range(len(self)):
            if self.iget_job(i) != other.iget_job(i):
                return False
        return True
