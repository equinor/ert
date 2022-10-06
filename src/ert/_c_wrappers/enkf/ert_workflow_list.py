import ctypes
import os
from typing import Iterator, List

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._c_wrappers.enkf.hook_workflow import HookWorkflow
from ert._c_wrappers.job_queue import Workflow, WorkflowJob, WorkflowJoblist
from ert._c_wrappers.util.substitution_list import SubstitutionList


def _to_c_string_arr(lst: List[str]):
    arr = (ctypes.c_char_p * len(lst))()
    arr[:] = [s.encode("utf-8") for s in lst]
    return arr


class ErtWorkflowList(BaseCClass):
    TYPE_NAME = "ert_workflow_list"
    _alloc = ResPrototype(
        "void* ert_workflow_list_alloc(subst_list, config_content, config_content)",
        bind=False,
    )
    _alloc_full = ResPrototype(
        "void* ert_workflow_list_alloc_full(subst_list, workflow_joblist)", bind=False
    )
    _free = ResPrototype("void ert_workflow_list_free(ert_workflow_list)")
    _alloc_namelist = ResPrototype(
        "stringlist_obj ert_workflow_list_alloc_namelist(ert_workflow_list)"
    )
    _has_workflow = ResPrototype(
        "bool ert_workflow_list_has_workflow(ert_workflow_list, char*)"
    )
    _get_workflow = ResPrototype(
        "workflow_ref ert_workflow_list_get_workflow(ert_workflow_list, char*)"
    )
    _add_workflow = ResPrototype(
        "workflow_ref ert_workflow_list_add_workflow(ert_workflow_list, char*, char*)"
    )
    _get_context = ResPrototype(
        "subst_list_ref ert_workflow_list_get_context(ert_workflow_list)"
    )
    _add_job = ResPrototype(
        "void ert_workflow_list_add_job(ert_workflow_list, char*, char*)"
    )
    _has_job = ResPrototype("bool ert_workflow_list_has_job(ert_workflow_list, char*)")
    _get_job = ResPrototype(
        "workflow_job_ref ert_workflow_list_get_job(ert_workflow_list, char*)"
    )
    _get_job_names = ResPrototype(
        "stringlist_obj ert_workflow_list_get_job_names(ert_workflow_list)"
    )

    _get_hook_workflow = ResPrototype(
        "hook_workflow_obj ert_workflow_list_iget_hook_workflow(ert_workflow_list, int)"
    )

    _num_hook_workflows = ResPrototype(
        "int ert_workflow_list_num_hook_workflows(ert_workflow_list)"
    )

    def __init__(
        self,
        subst_list=None,
        config_content=None,
        config_dict=None,
        site_config_content=None,
    ):
        if subst_list is None:
            raise ValueError(
                "Failed to construct ErtWorkflowList with no substitution list"
            )

        if config_content is None and config_dict is None:
            raise ValueError(
                "Failed to construct ErtWorkflowList instance with no config object"
            )

        if config_content is not None and config_dict is not None:
            raise ValueError(
                "Failed to construct ErtWorkflowList "
                "instance with multiple config object"
            )

        if config_content is not None and site_config_content is None:
            raise ValueError(
                "Failed to construct ErtWorkflowList instance. "
                "When using config_content also requires site_config_content"
            )

        c_ptr = None

        if config_content is not None:
            c_ptr = self._alloc(subst_list, config_content, site_config_content)

        if config_dict is not None:
            workflow_joblist = WorkflowJoblist()
            parser = WorkflowJob.configParser()
            for job in config_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, []):
                try:
                    new_job = WorkflowJob.fromFile(
                        config_file=job[ConfigKeys.PATH],
                        name=job[ConfigKeys.NAME],
                        parser=parser,
                    )
                except OSError:
                    print(f"WARNING: Unable to create job from {job[ConfigKeys.PATH]}")
                    continue
                if new_job is not None:
                    workflow_joblist.addJob(new_job)
                    new_job.convertToCReference(None)

            for job_path in config_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, []):
                if not os.path.isdir(job_path):
                    print(f"WARNING: Unable to open job directory {job_path}")
                    continue

                files = os.listdir(job_path)
                for file_name in files:
                    full_path = os.path.join(job_path, file_name)
                    try:
                        new_job = WorkflowJob.fromFile(
                            config_file=full_path, parser=parser
                        )
                        workflow_joblist.addJob(new_job)
                        new_job.convertToCReference(None)
                    except OSError:
                        print(f"WARNING: Unable to create job from {full_path}")
                        continue

            workflow_joblist.convertToCReference(None)

            # HOOK_WORKFLOW
            hook_workflow_names = []
            hook_workflow_run_modes = []
            if ConfigKeys.HOOK_WORKFLOW_KEY in config_dict:
                for hook_workflow_name, run_mode_name in config_dict[
                    ConfigKeys.HOOK_WORKFLOW_KEY
                ]:
                    if run_mode_name not in [
                        runtime.name for runtime in HookRuntime.enums()
                    ]:
                        raise ValueError(f"Run mode {run_mode_name} not supported")
                    hook_workflow_names.append(hook_workflow_name)
                    hook_workflow_run_modes.append(run_mode_name)

            c_ptr = self._alloc_full(
                subst_list,
                workflow_joblist,
                _to_c_string_arr(hook_workflow_names),
                _to_c_string_arr(hook_workflow_run_modes),
                len(hook_workflow_names),
            )

        if c_ptr is None:
            raise ValueError("Failed to construct ErtWorkflowList instance")

        super().__init__(c_ptr)

        if config_dict is not None:
            for job in config_dict.get(ConfigKeys.LOAD_WORKFLOW, []):
                self.addWorkflow(job[ConfigKeys.PATH], job[ConfigKeys.NAME])

    def getWorkflowNames(self) -> List[str]:
        return list(self._alloc_namelist())

    def __contains__(self, workflow_name: str) -> bool:
        assert isinstance(workflow_name, str)
        return self._has_workflow(workflow_name)

    def __getitem__(self, item) -> Workflow:
        if item not in self:
            raise KeyError(f"Item '{item}'  is not in the list of available workflows.")

        return self._get_workflow(item).setParent(self)

    def getContext(self) -> SubstitutionList:
        return self._get_context()

    def __str__(self):
        return f"ErtWorkflowList with jobs: {self.getJobNames()}"

    def addWorkflow(self, wf_name: str, wf_path: str):
        self._add_workflow(wf_name, wf_path).setParent(self)

    def addJob(self, job_name: str, job_path: str):
        self._add_job(job_name, job_path)

    def hasJob(self, job_name: str) -> bool:
        return self._has_job(job_name)

    def getJob(self, job_name: str) -> WorkflowJob:
        return self._get_job(job_name)

    def getJobNames(self) -> List[str]:
        return list(self._get_job_names())

    def getPluginJobs(self) -> List[WorkflowJob]:
        plugins = []
        for job_name in self.getJobNames():
            job = self.getJob(job_name)
            if job.isPlugin():
                plugins.append(job)
        return plugins

    def free(self):
        self._free()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        job_dicts = [
            {ConfigKeys.NAME: name, ConfigKeys.PATH: self.getJob(name).executable()}
            for name in self.getJobNames()
        ]
        return (
            "ErtWorkflowList(config_dict={"
            f"'{ConfigKeys.LOAD_WORKFLOW_JOB}': {job_dicts}, "
            "})"
        )

    def __eq__(self, other):
        if set(self.getJobNames()) != set(other.getJobNames()):
            return False

        for name_self, name_other in zip(
            sorted(self.getJobNames()), sorted(other.getJobNames())
        ):
            job_self = self.getJob(name_self)
            job_other = other.getJob(name_other)
            if job_self != job_other:
                return False

        if self.getWorkflowNames() != other.getWorkflowNames():
            return False

        for name_self, name_other in zip(
            self.getWorkflowNames(), other.getWorkflowNames()
        ):
            if self[name_self] != other[name_other]:
                return False

        if self._hook_workflows != other._hook_workflows:
            return False

        return True

    @property
    def _hook_workflows(self) -> List[HookWorkflow]:
        return [self._get_hook_workflow(i) for i in range(self._num_hook_workflows())]

    def get_workflows_hooked_at(self, run_time) -> Iterator[Workflow]:
        return map(
            lambda wh: wh.getWorkflow(),
            filter(lambda w: w.getRunMode() == run_time, self._hook_workflows),
        )
