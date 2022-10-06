from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ExtJoblist
from ert._c_wrappers.util.substitution_list import SubstitutionList


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
        "void forward_model_formatted_fprintf(forward_model, \
                                              char*, \
                                              char*, \
                                              char*, \
                                              subst_list, \
                                              env_varlist)"
    )

    def __init__(self, ext_joblist: ExtJoblist):
        c_ptr = self._alloc(ext_joblist)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Failed to construct forward model "
                f"from provided ext_joblist {ext_joblist}"
            )

    def __len__(self):
        return self._get_length()

    def joblist(self) -> StringList:
        """@rtype: StringList"""
        return self._alloc_joblist()

    def iget_job(self, index) -> ExtJob:
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
        self,
        run_id,
        path,
        data_root,
        global_args: SubstitutionList,
        env_varlist: EnvironmentVarlist,
    ):
        self._formatted_fprintf(run_id, path, data_root, global_args, env_varlist)

    def __repr__(self):
        return self._create_repr(f"joblist={self.joblist()}")

    def get_size(self):
        return len(self)

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        for i in range(len(self)):
            if self.iget_job(i) != other.iget_job(i):
                return False
        return True
