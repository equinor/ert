#  Copyright (C) 2017  Statoil ASA, Norway. 
#   
#  The file 'site_config.py' is part of ERT - Ensemble based Reservoir Tool. 
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
from res.job_queue import JobQueue, ExtJoblist
from ecl.util import StringList, Hash



class QueueConfig(BaseCClass):

    TYPE_NAME = "queue_config"

    _free                  = EnkfPrototype("void queue_config_free( queue_config )")
    _alloc_job_queue       = EnkfPrototype("job_queue_obj queue_config_alloc_job_queue( queue_config )")
    _alloc                 = EnkfPrototype("void* queue_config_alloc()" , bind      = False)

    def __init__(self):
        c_ptr = self._alloc()
        super(QueueConfig, self).__init__(c_ptr)

    def alloc_job_queue(self):
        return self._alloc_job_queue()

    def free(self):
        self._free()



