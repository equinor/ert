from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class RunArg(BaseCClass):
    TYPE_NAME = "run_arg"
    _alloc = ResPrototype(
        "void* run_arg_alloc(char*, \
                                                       enkf_fs, \
                                                       int, \
                                                       int, \
                                                       char*, \
                                                       char*)",
        bind=False,
    )
    _free = ResPrototype("void run_arg_free(run_arg)")
    _get_queue_index_safe = ResPrototype("int  run_arg_get_queue_index_safe(run_arg)")
    _set_queue_index = ResPrototype("void run_arg_set_queue_index(run_arg, int)")
    _is_submitted = ResPrototype("bool run_arg_is_submitted(run_arg)")
    _get_run_id = ResPrototype("char* run_arg_get_run_id(run_arg)")
    _get_runpath = ResPrototype("char* run_arg_get_runpath(run_arg)")
    _get_iter = ResPrototype("int run_arg_get_iter(run_arg)")
    _get_iens = ResPrototype("int run_arg_get_iens(run_arg)")
    _get_status = ResPrototype("int run_arg_get_run_status(run_arg)")
    _get_job_name = ResPrototype("char* run_arg_get_job_name(run_arg)")

    def __init__(
        self,
        run_id,
        sim_fs,
        iens,
        itr,
        run_path,
        job_name,
    ):
        c_ptr = self._alloc(
            run_id,
            sim_fs,
            iens,
            itr,
            run_path,
            job_name,
        )
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Not constructed properly")
        self._sim_fs = sim_fs

    def free(self):
        self._free()

    def set_queue_index(self, index):
        self._set_queue_index(index)

    def getQueueIndex(self) -> int:
        qi = self._get_queue_index_safe()
        if qi < 0:
            raise ValueError("Cannot get queue index before job is submitted.")
        return qi

    def isSubmitted(self) -> bool:
        return self._is_submitted()

    def __repr__(self):
        if self.isSubmitted():
            su = "submitted"
            qi = self.getQueueIndex()
        else:
            su = "not submitted"
            qi = "--"

        return f"RunArg(queue_index = {qi}, {su}) {self._ad_str()}"

    def get_run_id(self) -> str:
        return self._get_run_id()

    @property
    def runpath(self) -> str:
        return self._get_runpath()

    @property
    def iter_id(self) -> int:
        return self._get_iter()

    @property
    def iens(self) -> int:
        return self._get_iens()

    @property
    def run_status(self):
        return self._get_status()

    @property
    def job_name(self):
        return self._get_job_name()

    @property
    def sim_fs(self):
        return self._sim_fs
