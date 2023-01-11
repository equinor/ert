import logging
import os
from typing import Iterator, List

from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._c_wrappers.job_queue import Workflow, WorkflowJob

logger = logging.getLogger(__name__)


class ErtWorkflowList:
    def __init__(
        self,
        workflow_job_info: list = None,
        workflow_job_dir_info: list = None,
        hook_workflow_info: list = None,
        workflow_info: list = None,
    ):
        workflow_job_info = [] if workflow_job_info is None else workflow_job_info
        workflow_job_dir_info = (
            [] if workflow_job_dir_info is None else workflow_job_dir_info
        )
        hook_workflow_info = [] if hook_workflow_info is None else hook_workflow_info
        workflow_info = [] if workflow_info is None else workflow_info

        self._workflow_jobs = {}
        self._workflow = {}
        self._hook_workflow_list = []
        self._parser = WorkflowJob.configParser()

        for workflow_job in workflow_job_info:
            self._add_workflow_job(workflow_job)

        for job_path in workflow_job_dir_info:
            self._add_workflow_job_dir(job_path)

        for work in workflow_info:
            self._add_workflow(work)

        for hook_name, mode_name in hook_workflow_info:
            self._add_hook_workflow(hook_name, mode_name)

    @classmethod
    def from_dict(cls, content_dict) -> "ErtWorkflowList":
        workflow_job_info = []
        workflow_job_dir_info = []
        hook_workflow_info = []
        workflow_info = []

        for workflow in content_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, []):
            workflow_job_info.append(workflow)

        for workflow_dir in content_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, []):
            workflow_job_dir_info.append(workflow_dir)

        for name, mode in content_dict.get(ConfigKeys.HOOK_WORKFLOW_KEY, []):
            hook_workflow_info.append([name, mode])

        for workflow in content_dict.get(ConfigKeys.LOAD_WORKFLOW, []):
            workflow_info.append(workflow)

        config = cls(
            workflow_job_info=workflow_job_info,
            workflow_job_dir_info=workflow_job_dir_info,
            hook_workflow_info=hook_workflow_info,
            workflow_info=workflow_info,
        )

        return config

    def getWorkflowNames(self) -> List[str]:
        return list(self._workflow.keys())

    def getWorkflowPath(self, workflow_name: str) -> str:
        return self._workflow[workflow_name].src_file

    def __contains__(self, workflow_name: str) -> bool:
        return workflow_name in self._workflow

    def __getitem__(self, item) -> Workflow:
        if item not in self:
            raise KeyError(f"Item '{item}' is not in the list of available workflows.")

        return self._workflow[item]

    def __str__(self):
        return f"ErtWorkflowList with jobs: {self.getJobNames()}"

    def hasJob(self, job_name: str) -> bool:
        return job_name in self._workflow_jobs

    def getJob(self, job_name: str) -> WorkflowJob:
        return self._workflow_jobs[job_name]

    def getJobNames(self) -> List[str]:
        return list(self._workflow_jobs.keys())

    def getPluginJobs(self) -> List[WorkflowJob]:
        plugins = []
        for job_name in self.getJobNames():
            job = self.getJob(job_name)
            if job.isPlugin():
                plugins.append(job)
        return plugins

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        job_dicts = [
            {ConfigKeys.NAME: name, ConfigKeys.PATH: self.getJob(name).executable()}
            for name in self.getJobNames()
        ]

        workflow_dicts = [
            {ConfigKeys.NAME: workflow, ConfigKeys.PATH: self.getWorkflowPath(workflow)}
            for workflow in self.getWorkflowNames()
        ]

        hook_dicts = [
            {ConfigKeys.NAME: name, ConfigKeys.RUNMODE: mode}
            for name, mode in self._hook_workflow_list
        ]

        return (
            "ErtWorkflowList(config_dict={"
            f"'{ConfigKeys.LOAD_WORKFLOW_JOB}': {job_dicts}, "
            f"'{ConfigKeys.LOAD_WORKFLOW}': {workflow_dicts}, "
            f"'{ConfigKeys.HOOK_WORKFLOW_KEY}': {hook_dicts}, "
            "})"
        )

    def __eq__(self, other):
        if set(self.getJobNames()) != set(other.getJobNames()):
            return False

        for name_self, name_other in zip(
            sorted(self.getJobNames()), sorted(other.getJobNames())
        ):
            if self.getJob(name_self) != other.getJob(name_other):
                return False

        if set(self.getWorkflowNames()) != set(other.getWorkflowNames()):
            return False

        for name_self, name_other in zip(
            sorted(self.getWorkflowNames()), sorted(other.getWorkflowNames())
        ):
            if self[name_self] != other[name_other]:
                return False

        if self._hook_workflows != other._hook_workflows:
            return False

        for mode in HookRuntime.enums():
            for name_self, name_other in zip(
                sorted(self.get_workflows_hooked_at(mode)),
                sorted(other.get_workflows_hooked_at(mode)),
            ):
                if name_self != name_other:
                    return False
                if name_self.src_file != name_other.src_file:
                    return False

        return True

    def _add_workflow_job(self, workflow_job):
        try:
            new_job = WorkflowJob.fromFile(
                config_file=workflow_job[0],
                name=None if len(workflow_job) == 1 else workflow_job[1],
                parser=self._parser,
            )
            if new_job is not None:
                self._workflow_jobs[new_job.name()] = new_job
                new_job.convertToCReference(None)
                logger.info(f"Adding workflow job:{new_job.name()}")

        except OSError:
            print(f"WARNING: Unable to create job from {workflow_job[0]}")

    def _add_workflow_job_dir(self, job_path):
        if not os.path.isdir(job_path):
            print(f"WARNING: Unable to open job directory {job_path}")
            return

        files = os.listdir(job_path)
        for file_name in files:
            full_path = os.path.join(job_path, file_name)
            self._add_workflow_job([full_path])

    def _add_workflow(self, work):
        filename = os.path.basename(work[0]) if len(work) == 1 else work[1]
        self._workflow[filename] = Workflow(work[0], self._workflow_jobs)

    def _add_hook_workflow(self, hook_name, mode_name):
        if mode_name not in [runtime.name for runtime in HookRuntime.enums()]:
            raise ValueError(f"Run mode {mode_name} not supported")

        if hook_name not in self._workflow:
            raise ConfigValidationError(
                errors=[f"Cannot setup hook for non-existing job name {hook_name}"]
            )

        self._hook_workflow_list.append((hook_name, mode_name))

    @property
    def _hook_workflows(self) -> List[Workflow]:
        return [self._workflow[name] for name, _ in self._hook_workflow_list]

    def get_workflows_hooked_at(self, run_time: HookRuntime) -> Iterator[Workflow]:
        return map(
            lambda hookname: self._workflow[hookname[0]],
            filter(lambda hook: hook[1] == run_time.name, self._hook_workflow_list),
        )
