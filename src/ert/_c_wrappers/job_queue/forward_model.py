import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ExtJoblist
from ert._clib import job_kw

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.substituter import Substituter

logger = logging.getLogger(__name__)


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

    def __init__(self, ext_joblist: ExtJoblist):
        c_ptr = self._alloc(ext_joblist)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Failed to construct forward model "
                f"from provided ext_joblist {ext_joblist}"
            )

    def __len__(self) -> int:
        return self._get_length()

    def job_name_list(self) -> List[str]:
        return list(self._alloc_joblist())

    def iget_job(self, index) -> ExtJob:
        return self._iget_job(index).setParent(self)

    def add_job(self, name: str) -> ExtJob:
        return self._add_job(name).setParent(self)

    def __iter__(self) -> Iterator[ExtJob]:
        return iter([self.iget_job(idx) for idx in range(len(self))])

    def clear(self):
        self._clear()

    def free(self):
        self._free()

    def formatted_fprintf(
        self,
        run_id,
        path,
        data_root,
        iens: int,
        itr: int,
        substituter: "Substituter",
        env_varlist: EnvironmentVarlist,
        filename: str = "jobs.json",
    ):
        def substitute(job, string):
            if string is not None:
                return substituter.substitute(
                    job.get_private_args().substitute(string), iens, itr
                )
            else:
                return string

        def positive_or_null_int(integer: int) -> Optional[int]:
            return integer if integer > 0 else None

        def handle_default(job: ExtJob, arg: str) -> str:
            return job.get_default_mapping().get(arg, arg)

        def filter_env_dict(job, d):
            result = {}
            for key, value in d.items():
                new_key = substitute(job, key)
                new_value = substitute(job, value)
                if new_value is None:
                    result[new_key] = None
                elif not (new_value[0] == "<" and new_value[-1] == ">"):
                    # Remove values containing "<XXX>". These are expected to be
                    # replaced by substitute, but were not.
                    result[new_key] = new_value
                else:
                    logger.warning(
                        "Environment variable %s skipped due to unmatched define %s",
                        new_key,
                        new_value,
                    )
            # Its expected that empty dicts be replaced with "null"
            # in jobs.json
            if not result:
                return None
            return result

        with open(Path(path) / filename, mode="w", encoding="utf-8") as fptr:
            json.dump(
                {
                    "DATA_ROOT": data_root,
                    "global_environment": env_varlist.varlist,
                    "global_update_path": env_varlist.updatelist,
                    "jobList": [
                        {
                            "name": substitute(job, job.name()),
                            "executable": substitute(job, job.get_executable()),
                            "target_file": substitute(job, job.get_target_file()),
                            "error_file": substitute(job, job.get_error_file()),
                            "start_file": substitute(job, job.get_start_file()),
                            "stdout": substitute(job, job.get_stdout_file())
                            + f".{idx}",
                            "stderr": substitute(job, job.get_stderr_file())
                            + f".{idx}",
                            "stdin": substitute(job, job.get_stdin_file()),
                            "argList": [
                                handle_default(job, substitute(job, arg))
                                for arg in job.get_arglist()
                            ],
                            "environment": filter_env_dict(job, job.get_environment()),
                            "exec_env": filter_env_dict(job, job.get_exec_env()),
                            "license_path": substitute(job, job.get_license_path()),
                            "max_running_minutes": positive_or_null_int(
                                job.get_max_running_minutes()
                            ),
                            "max_running": positive_or_null_int(job.get_max_running()),
                            "min_arg": positive_or_null_int(job.min_arg),
                            "arg_types": [
                                job_kw.kw_from_type(typ) for typ in job.arg_types
                            ],
                            "max_arg": positive_or_null_int(job.max_arg),
                        }
                        for idx, job in enumerate(self)
                    ],
                    "run_id": run_id,
                    "ert_pid": str(os.getpid()),
                },
                fptr,
            )

    def __repr__(self) -> str:
        return self._create_repr(f"joblist={self.job_name_list()}")

    def get_size(self) -> int:
        return len(self)

    def __ne__(self, other) -> bool:
        return not self == other

    def __eq__(self, other) -> bool:
        for i in range(len(self)):
            if self.iget_job(i) != other.iget_job(i):
                return False
        return True
