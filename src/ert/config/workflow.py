from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .parsing import ConfigValidationError, ErrorInfo, init_workflow_schema, parse
from .parsing.types import Defines
from .workflow_job import _WorkflowJob


@dataclass
class Workflow:
    src_file: str
    cmd_list: list[tuple[_WorkflowJob, Any]]

    def __len__(self) -> int:
        return len(self.cmd_list)

    def __getitem__(self, index: int) -> tuple[_WorkflowJob, Any]:
        return self.cmd_list[index]

    def __iter__(self) -> Iterator[tuple[_WorkflowJob, Any]]:
        return iter(self.cmd_list)

    @classmethod
    def _parse_command_list(
        cls,
        src_file: str,
        context: Defines,
        job_dict: dict[str, _WorkflowJob],
    ) -> list[tuple[_WorkflowJob, Any]]:
        schema = init_workflow_schema()
        config_dict = parse(src_file, schema, pre_defines=context)

        parsed_workflow_job_names = config_dict.keys() - {"DEFINE"}

        all_workflow_jobs = []
        errors = []

        for job_name in parsed_workflow_job_names:
            for instructions in config_dict[job_name]:  # type: ignore
                job_name_with_context = instructions.keyword_token  # type: ignore
                job = job_dict.get(job_name)
                if job is None:
                    errors.append(
                        ErrorInfo(
                            f"Job with name: {job_name} is not recognized"
                        ).set_context(job_name_with_context)
                    )
                    continue
                elif job.min_args is not None and job.min_args > len(instructions):
                    errors.append(
                        ErrorInfo(
                            f"Job with name: {job_name} does not have enough arguments,"
                            f" expected at least: {job.min_args}, got: {instructions}"
                        ).set_context(job_name_with_context)
                    )
                    continue
                elif job.max_args is not None and job.max_args < len(instructions):
                    errors.append(
                        ErrorInfo(
                            f"Job with name: {job_name} has too many arguments, "
                            f"expected at most: {job.min_args}, got: {instructions}"
                        ).set_context(job_name_with_context)
                    )
                    continue

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
        job_dict: dict[str, _WorkflowJob],
    ) -> Workflow:
        cmd_list = cls._parse_command_list(
            src_file=src_file,
            context=list(map(list, context.items())) if context else [],
            job_dict=job_dict,
        )

        return cls(src_file, cmd_list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
