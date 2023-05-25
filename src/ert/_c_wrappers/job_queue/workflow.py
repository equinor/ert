from __future__ import annotations

import os
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from ert._c_wrappers.config import ConfigParser, UnrecognizedEnum
from ert.parsing import ConfigValidationError, init_workflow_schema, lark_parse
from ert.parsing.error_info import ErrorInfo

if TYPE_CHECKING:
    from ert._c_wrappers.job_queue import WorkflowJob
    from ert._c_wrappers.util import SubstitutionList


def _workflow_parser(workflow_jobs: Dict[str, WorkflowJob]) -> ConfigParser:
    parser = ConfigParser()
    for name, job in workflow_jobs.items():
        item = parser.add(name)
        item.set_argc_minmax(job.min_args, job.max_args)
        for i, t in enumerate(job.arg_types):
            item.iset_type(i, t)
    return parser


class Workflow:
    def __init__(
        self,
        src_file: str,
        cmd_list: List[Tuple[WorkflowJob, Any]],
    ):
        self.src_file = src_file
        self.cmd_list = cmd_list

    def __len__(self):
        return len(self.cmd_list)

    def __getitem__(self, index: int) -> Tuple[WorkflowJob, Any]:
        return self.cmd_list[index]

    def __iter__(self) -> Iterator[Tuple[WorkflowJob, Any]]:
        return iter(self.cmd_list)

    @classmethod
    def _parse_command_list_with_old_parser(
        cls, src_file: str, job_dict: Dict[str, WorkflowJob]
    ):
        parser = _workflow_parser(job_dict)
        try:
            content = parser.parse(
                src_file, unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_ERROR
            )
        except ConfigValidationError as err:
            err.config_file = src_file
            raise err from None

        cmd_list = []
        for line in content:
            cmd_list.append(
                (
                    job_dict[line.get_kw()],
                    [line.igetString(i) for i in range(len(line))],
                )
            )

        return cmd_list

    @classmethod
    def _parse_command_list_with_new_parser(
        cls, src_file: str, job_dict: Dict[str, WorkflowJob]
    ):
        schema = init_workflow_schema()
        parsed = lark_parse(src_file, schema, pre_defines=[])

        workflow_names = parsed.keys() - {"DEFINE"}

        all_workflows = []
        errors = []

        for name in workflow_names:
            for args in parsed[name]:
                workflow_job_name = args.keyword_token

                if workflow_job_name not in job_dict:
                    errors.append(
                        ErrorInfo(
                            filename=src_file,
                            message=f"Job with name: {workflow_job_name}"
                            f" is not recognized",
                        ).set_context(workflow_job_name)
                    )
                    continue

                all_workflows.append((workflow_job_name, args))

        if errors:
            raise ConfigValidationError.from_collected(errors)

        # Order matters, so we need to sort it
        # by the line attached to the context token
        all_workflows.sort(key=lambda x: x[0].line)

        def insert_job(workflow: Tuple[str, List[str]]):
            my_name, my_args = workflow
            return job_dict[my_name], my_args

        command_list = list(map(insert_job, all_workflows))

        return command_list

    @classmethod
    def from_file(
        cls,
        src_file: str,
        context: Optional[SubstitutionList],
        job_dict: Dict[str, WorkflowJob],
        use_new_parser: bool = True,
    ):
        to_compile = src_file
        if not os.path.exists(src_file):
            raise ConfigValidationError(
                f"Workflow file {src_file} does not exist", config_file=src_file
            )
        if context is not None:
            tmpdir = mkdtemp("ert_workflow")
            to_compile = os.path.join(tmpdir, "ert-workflow")
            context.substitute_file(src_file, to_compile)

        cmd_list = (
            cls._parse_command_list_with_new_parser(
                src_file=to_compile, job_dict=job_dict
            )
            if use_new_parser
            else cls._parse_command_list_with_old_parser(
                src_file=to_compile, job_dict=job_dict
            )
        )

        return cls(src_file, cmd_list)

    def __eq__(self, other):
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
