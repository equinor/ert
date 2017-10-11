#  Copyright (C) 2014  Statoil ASA, Norway. 
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
from res.enkf import EnkfPrototype

class RunArg(BaseCClass):
    TYPE_NAME = "run_arg"

    _alloc_ENSEMBLE_EXPERIMENT = EnkfPrototype("run_arg_obj run_arg_alloc_ENSEMBLE_EXPERIMENT(char*, enkf_fs, int, int, char*)", bind = False)
    _free                      = EnkfPrototype("void run_arg_free(run_arg)")
    _get_queue_index           = EnkfPrototype("int  run_arg_get_queue_index(run_arg)")
    _is_submitted              = EnkfPrototype("bool run_arg_is_submitted(run_arg)")
    _get_run_id                = EnkfPrototype("char* run_arg_get_run_id(run_arg)")
    
    def __init__(self):
        raise NotImplementedError("Cannot instantiat RunArg directly!")

    @classmethod
    def createEnsembleExperimentRunArg(cls, run_id, fs, iens, runpath, iter=0):
        return cls._alloc_ENSEMBLE_EXPERIMENT(run_id, fs, iens, iter, runpath)

    def free(self):
        self._free()

    def getQueueIndex(self):
        return self._get_queue_index()

    def isSubmitted(self):
        return self._is_submitted()

    def __repr__(self):
        su = 'submitted' if self.isSubmitted() else 'not submitted'
        qi = self.getQueueIndex()
        return 'RunArg(queue_index = %d, %s) %s' % (qi, su, self._ad_str())


    def get_run_id(self):
        return self._get_run_id( )
