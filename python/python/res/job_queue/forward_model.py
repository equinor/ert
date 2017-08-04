#  Copyright (C) 2012  Statoil ASA, Norway. 
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
from res.job_queue import ExtJob, QueuePrototype, ExtJoblist
from ecl.util import StringList
from res.util.substitution_list import SubstitutionList


class ForwardModel(BaseCClass):
    TYPE_NAME      = "forward_model"

    _alloc         = QueuePrototype("void* forward_model_alloc(ext_joblist)", bind=False)
    _free          = QueuePrototype("void forward_model_free( forward_model )")
    _clear         = QueuePrototype("void forward_model_clear(forward_model)")
    _add_job       = QueuePrototype("ext_job_ref forward_model_add_job(forward_model, char*)")
    _alloc_joblist = QueuePrototype("stringlist_obj forward_model_alloc_joblist(forward_model)")
    _iget_job      = QueuePrototype("ext_job_ref forward_model_iget_job( forward_model, int)")
    _formatted_fprintf = QueuePrototype("void forward_model_formatted_fprintf(forward_model, char*, char*, subst_list, int)")

    def __init__(self, ext_joblist):
        c_ptr = self._alloc(ext_joblist)
        if c_ptr:
            super(ForwardModel, self).__init__(c_ptr)
        else:
            raise ValueError('Failed to construct forward model from provided ext_joblist %s' % ext_joblist)

    def joblist(self):
        """ @rtype: StringList """
        return self._alloc_joblist( )

    def iget_job(self, index):
        """ @rtype: ExtJob """
        return self._iget_job(index).setParent(self)

    def add_job(self, name):
        """ @rtype: ExtJob """
        return self._add_job(name).setParent(self)

    def clear(self):
        self._clear( )

    def free(self):
        self._free( )

    def formatted_fprintf(self, run_id, path, global_args, umask):
        self._formatted_fprintf(run_id, path, global_args, umask)

    def __repr__(self):
        return self._create_repr('joblist=%s' % self.joblist())
