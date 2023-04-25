from __future__ import annotations

import os
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from ert._c_wrappers.config import ConfigParser, UnrecognizedEnum
from ert.parsing import ConfigValidationError

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
    def from_file(
        cls,
        src_file: str,
        context: Optional[SubstitutionList],
        job_list: Dict[str, WorkflowJob],
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

        cmd_list = []
        parser = _workflow_parser(job_list)
        try:
            content = parser.parse(
                to_compile, unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_ERROR
            )
        except ConfigValidationError as err:
            err.config_file = src_file
            raise err from None

        for line in content:
            cmd_list.append(
                (
                    job_list[line.get_kw()],
                    [line.igetString(i) for i in range(len(line))],
                )
            )

        return cls(src_file, cmd_list)

    def __eq__(self, other):
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
