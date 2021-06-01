#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ext_joblist.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.job_queue import ExtJob
from ecl.util.util import StringList


class ExtJoblist(BaseCClass):
    TYPE_NAME = "ext_joblist"
    _alloc = ResPrototype("void* ext_joblist_alloc( )", bind=False)
    _free = ResPrototype("void ext_joblist_free( ext_joblist )")
    _alloc_list = ResPrototype("stringlist_ref ext_joblist_alloc_list(ext_joblist)")
    _get_job = ResPrototype("ext_job_ref ext_joblist_get_job(ext_joblist, char*)")
    _del_job = ResPrototype("int ext_joblist_del_job(ext_joblist, char*)")
    _has_job = ResPrototype("int ext_joblist_has_job(ext_joblist, char*)")
    _add_job = ResPrototype("void ext_joblist_add_job(ext_joblist, char*, ext_job)")
    _get_jobs = ResPrototype("hash_ref ext_joblist_get_jobs(ext_joblist)")
    _size = ResPrototype("int ext_joblist_get_size(ext_joblist)")

    def __init__(self):
        c_ptr = self._alloc()
        super(ExtJoblist, self).__init__(c_ptr)

    def get_jobs(self):
        """@rtype: Hash"""
        jobs = self._get_jobs()
        jobs.setParent(self)
        return jobs

    def __len__(self):
        return self._size()

    def __contains__(self, job):
        return self._has_job(job)

    def __iter__(self):
        names = self.getAvailableJobNames()
        for job in names:
            yield self[job]

    def __getitem__(self, job):
        if job in self:
            return self._get_job(job).setParent(self)

    def getAvailableJobNames(self):
        """@rtype: StringList"""
        return [str(x) for x in self._alloc_list().setParent(self)]

    def del_job(self, job):
        return self._del_job(job)

    def has_job(self, job):
        return job in self

    def get_job(self, job):
        """@rtype: ExtJob"""
        return self[job]

    def add_job(self, job_name, new_job):
        if not new_job.isReference():
            new_job.convertToCReference(self)

        self._add_job(job_name, new_job)

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr("size=%d, joblist=%s" % (len(self), self.get_jobs()))
