from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from ert.base_model_context import BaseModelWithContextSupport

from .parsing import ConfigValidationError, init_workflow_schema, parse
from .parsing.types import Defines
from .workflow_job import BaseErtScriptWorkflow, WorkflowJob


class Workflow(BaseModelWithContextSupport):
    src_file: str
    cmd_list: list[tuple[WorkflowJob, Any]]

    def __len__(self) -> int:
        return len(self.cmd_list)

    def __getitem__(self, index: int) -> tuple[WorkflowJob, Any]:
        return self.cmd_list[index]

    def __iter__(self) -> Iterator[tuple[WorkflowJob, Any]]:  # type: ignore
        return iter(self.cmd_list)

    @staticmethod
    def validate_workflow_job(
        job_name: str,
        args: list[Any],
        job_dict: dict[str, WorkflowJob],
    ) -> WorkflowJob:
        job = job_dict.get(job_name)

        if job is None:
            raise ConfigValidationError.with_context(
                f"Job with name: {job_name} is not recognized", job_name
            )

        if job.min_args is not None and job.min_args > len(args):
            raise ConfigValidationError.with_context(
                f"Job with name: {job_name} does not have enough"
                f" arguments, expected at least: {job.min_args}, got: {args}",
                job_name,
            )

        if job.max_args is not None and job.max_args < len(args):
            raise ConfigValidationError.with_context(
                f"Job with name: {job_name} has too many arguments, "
                f"expected at most: {job.max_args}, got: {args}",
                job_name,
            )

        if isinstance(job, BaseErtScriptWorkflow):
            job.load_ert_script_class().validate(args)

        return job

    @classmethod
    def _parse_command_list(
        cls,
        src_file: str,
        context: Defines,
        job_dict: dict[str, WorkflowJob],
    ) -> list[tuple[WorkflowJob, Any]]:
        schema = init_workflow_schema()
        config_dict = parse(src_file, schema, pre_defines=context)

        parsed_workflow_job_names = config_dict.keys() - {"DEFINE"}

        all_workflow_jobs = []
        errors = []

        for job_name in parsed_workflow_job_names:
            for instructions in config_dict[job_name]:
                job_name_with_context = instructions.token
                try:
                    cls.validate_workflow_job(
                        job_name_with_context, instructions, job_dict
                    )
                except ConfigValidationError as err:
                    errors.append(err)

                all_workflow_jobs.append((job_name_with_context, instructions))

        if errors:
            raise ConfigValidationError.from_collected(errors)

        # Order matters, so we need to sort it
        # by the line attached to the context token
        all_workflow_jobs.sort(key=lambda x: x[0].line)

        return [
            (job_dict[name], instructions) for (name, instructions) in all_workflow_jobs
        ]

    @classmethod
    def from_file(
        cls,
        src_file: str,
        context: dict[str, str] | None,
        job_dict: dict[str, WorkflowJob],
    ) -> Workflow:
        cmd_list = cls._parse_command_list(
            src_file=src_file,
            context=list(map(list, context.items())) if context else [],
            job_dict=job_dict,
        )

        return cls(src_file=src_file, cmd_list=cmd_list)

    @classmethod
    def from_instructions(
        cls,
        workflow_name: str,
        job_name: str,
        args: list[Any],
        job_dict: dict[str, WorkflowJob],
    ) -> Workflow:
        job = cls.validate_workflow_job(job_name, args, job_dict)
        return cls(
            src_file=workflow_name,
            cmd_list=[(job, args)],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
