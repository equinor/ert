from typing import Dict, Iterator, List

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue import ExtJob


class ExtJoblist(BaseCClass):
    TYPE_NAME = "ext_joblist"
    _alloc = ResPrototype("void* ext_joblist_alloc( )", bind=False)
    _free = ResPrototype("void ext_joblist_free( ext_joblist )")
    _alloc_list = ResPrototype("stringlist_ref ext_joblist_alloc_list(ext_joblist)")
    _get_job = ResPrototype("ext_job_ref ext_joblist_get_job(ext_joblist, char*)")
    _del_job = ResPrototype("int ext_joblist_del_job(ext_joblist, char*)")
    _has_job = ResPrototype("int ext_joblist_has_job(ext_joblist, char*)")
    _add_job = ResPrototype("void ext_joblist_add_job(ext_joblist, char*, ext_job)")
    _size = ResPrototype("int ext_joblist_get_size(ext_joblist)")

    def __init__(self):
        c_ptr = self._alloc()
        super().__init__(c_ptr)

    def get_jobs(self) -> Dict[str, ExtJob]:
        return {
            job_name: self.get_job(job_name) for job_name in self.getAvailableJobNames()
        }

    def __len__(self) -> int:
        return self._size()

    def __contains__(self, job_name: str) -> bool:
        return self._has_job(job_name)

    def __iter__(self) -> Iterator[ExtJob]:
        names = self.getAvailableJobNames()
        for job in names:
            yield self[job]

    def __getitem__(self, job_name: str) -> ExtJob:
        if job_name in self:
            return self._get_job(job_name).setParent(self)
        raise IndexError(f"Unknown job {job_name}")

    def getAvailableJobNames(self) -> List[str]:
        return [str(x) for x in self._alloc_list().setParent(self)]

    def del_job(self, job_name: str):
        return self._del_job(job_name)

    def has_job(self, job_name: str) -> bool:
        return job_name in self

    def get_job(self, job_name: str) -> ExtJob:
        return self[job_name]

    def add_job(self, job_name: str, new_job: ExtJob):
        if not new_job.isReference():
            new_job.convertToCReference(self)

        self._add_job(job_name, new_job)

    def free(self):
        self._free()

    def __repr__(self) -> str:
        return self._create_repr(f"size={len(self)}, joblist={self.get_jobs()}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtJoblist):
            return False
        return self.get_jobs() == other.get_jobs()
