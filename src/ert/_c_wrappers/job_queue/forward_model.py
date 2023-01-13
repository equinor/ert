import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob
from ert._clib import job_kw

if TYPE_CHECKING:
    from ert._c_wrappers.util.substitution_list import SubstitutionList

logger = logging.getLogger(__name__)


@dataclass
class ForwardModel:
    jobs: List[ExtJob]

    def job_name_list(self) -> List[str]:
        return [j.name for j in self.jobs]

    def formatted_fprintf(
        self,
        run_id,
        path,
        data_root,
        iens: int,
        itr: int,
        context: "SubstitutionList",
        env_varlist: EnvironmentVarlist,
        filename: str = "jobs.json",
    ):
        with open(Path(path) / filename, mode="w", encoding="utf-8") as fptr:
            json.dump(
                self.get_job_data(
                    run_id,
                    data_root,
                    iens,
                    itr,
                    context,
                    env_varlist,
                ),
                fptr,
            )

    def get_job_data(
        self,
        run_id,
        data_root,
        iens: int,
        itr: int,
        context: "SubstitutionList",
        env_varlist: EnvironmentVarlist,
    ) -> Dict[str, Any]:
        def substitute(job, string):
            if string is not None:
                copy_private_args = copy.deepcopy(job.private_args)
                # We need to still be able to handle that the user has
                # written e.g. FORWARD_MODEL JOB(<ITER>=<ITER>). The
                # way this argument passing works is at odds with expected
                # behavior, and is a known bug/problem.

                if (
                    "<ITER>" not in copy_private_args
                    or copy_private_args["<ITER>"] == "<ITER>"
                ):
                    copy_private_args.addItem("<ITER>", str(itr))
                if (
                    "<IENS>" not in copy_private_args
                    or copy_private_args["<IENS>"] == "<IENS>"
                ):
                    copy_private_args.addItem("<IENS>", str(iens))
                string = copy_private_args.substitute(string)
                return context.substitute_real_iter(string, iens, itr)
            else:
                return string

        def handle_default(job: ExtJob, arg: str) -> str:
            return job.default_mapping.get(arg, arg)

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

        return {
            "DATA_ROOT": data_root,
            "global_environment": env_varlist.varlist,
            "global_update_path": env_varlist.updatelist,
            "jobList": [
                {
                    "name": substitute(job, job.name),
                    "executable": substitute(job, job.executable),
                    "target_file": substitute(job, job.target_file),
                    "error_file": substitute(job, job.error_file),
                    "start_file": substitute(job, job.start_file),
                    "stdout": substitute(job, job.stdout_file) + f".{idx}"
                    if job.stdout_file
                    else None,
                    "stderr": substitute(job, job.stderr_file) + f".{idx}"
                    if job.stderr_file
                    else None,
                    "stdin": substitute(job, job.stdin_file),
                    "argList": [
                        handle_default(job, substitute(job, arg)) for arg in job.arglist
                    ],
                    "environment": filter_env_dict(job, job.environment),
                    "exec_env": filter_env_dict(job, job.exec_env),
                    "max_running_minutes": job.max_running_minutes,
                    "max_running": job.max_running,
                    "min_arg": job.min_arg,
                    "arg_types": [job_kw.kw_from_type(typ) for typ in job.arg_types],
                    "max_arg": job.max_arg,
                }
                for idx, job in enumerate(self.jobs)
            ],
            "run_id": run_id,
            "ert_pid": str(os.getpid()),
        }
