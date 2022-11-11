import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob
from ert._clib import job_kw

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.substituter import Substituter

logger = logging.getLogger(__name__)


@dataclass
class ForwardModel:
    jobs: List[ExtJob]

    def job_name_list(self) -> List[str]:
        return [j.name() for j in self.jobs]

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
                        for idx, job in enumerate(self.jobs)
                    ],
                    "run_id": run_id,
                    "ert_pid": str(os.getpid()),
                },
                fptr,
            )
