from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from typing_extensions import NotRequired, Unpack

from ert.config.parsing.config_errors import ConfigWarning
from ert.substitution_list import SubstitutionList

from .parsing import (
    SchemaItemType,
)

logger = logging.getLogger(__name__)


class ForwardModelStepValidationError(Exception):
    """Thrown when the user calls the forward model incorrectly.

    Can be subtyped by the implementation of ForwardModelStepPlugin and
    thrown from `validate_pre_realization_run` or `validate_pre_experiment`.
    """


class ForwardModelStepWarning(ConfigWarning):
    pass


class ForwardModelStepJSON(TypedDict):
    """
    A dictionary containing information about how a forward model step should be run
    on the queue.

    Attributes:
        executable: The name of the executable to be run
        target_file: Name of file expected to be produced after the forward
            model step is executed successfully.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        error_file: Name of file expected to be produced after a forward
            model step errors.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        start_file: Name of file expected to be produced upon start of
            a forward model step's execution.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        stdout: File where this forward model step's stdout is written
        stderr: File where this forward model step's stderr is written
        stdin: File where this forward model step's stdin is written
        argList: List of command line arguments to be given to the executable.
        environment: Dictionary of environment variables to inject into the
            environment of the forward model step run
        exec_env: Dictionary of environment variables to inject into the execution
            environment of RMS
        max_running_minutes: Maximum runtime in minutes. If the forward model step
            takes longer than this, the job is requested to be cancelled.
    """

    name: str
    executable: list[str]
    target_file: str
    error_file: str
    start_file: str
    stdout: str
    stderr: str
    stdin: str
    argList: list[str]
    environment: dict[str, str]
    exec_env: dict[str, str]
    max_running_minutes: int


class ForwardModelStepOptions(TypedDict, total=False):
    stdin_file: NotRequired[str]
    stdout_file: NotRequired[str]
    stderr_file: NotRequired[str]
    start_file: NotRequired[str]
    target_file: NotRequired[str]
    error_file: NotRequired[str]
    max_running_minutes: NotRequired[int]
    environment: NotRequired[dict[str, Union[str, int]]]
    exec_env: NotRequired[dict[str, Union[str, int]]]
    default_mapping: NotRequired[dict[str, Union[str, int]]]


@dataclass
class ForwardModelStepDocumentation:
    config_file: Optional[str] = field(default=None)
    source_package: str = field(default="ert")
    source_function_name: str = field(default="ert")
    description: str = field(default="No description")
    examples: str = field(default="No examples")
    category: Union[
        Literal[
            "utility.file_system",
            "simulators.reservoir",
            "modelling.reservoir",
            "utility.templating",
        ],
        str,
    ] = field(default="Uncategorized")


@dataclass
class ForwardModelStep:
    """
    Holds information to execute one step of a forward model

    Attributes:
        executable: The name of the executable to be run
        stdin_file: File where this forward model step's stdout is written
        stdout_file: File where this forward model step's stderr is written
        stderr_file: File where this forward model step's stdin is written
        target_file: Name of file expected to be produced after the forward
            model step is executed successfully.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        error_file: Name of file expected to be produced after a forward
            model step errors.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        start_file: Name of file expected to be produced upon start of
            a forward model step's execution.
            This file is used for the legacy ERT queue driver, and may be deprecated.
        max_running_minutes: Maximum runtime in minutes. If the forward model step
            takes longer than this, the job is requested to be cancelled.
        min_arg: The minimum number of arguments
        max_arg: The maximum number of arguments
        arglist: The arglist with which the executable is invoked
        required_keywords: Keywords that are required to be supplied by the user
        arg_types: Types of user-provided arguments, this list must have indices
            corresponding to the number of args that are user specified
            and thus also subject to substitution.
        environment: Dictionary representing environment variables to inject into the
            environment of the forward model step run
        exec_env: Dictionary of environment variables to inject into the execution
            environment of RMS
        default_mapping: Default values for optional arguments provided by the user.
            For example { "A": "default_A" }
        private_args: A dictionary of user-provided keyword arguments.
            For example, if the user provides <A>=2, the dictionary will contain
            { "A": "2" }
        help_text: Some meaningless old code we don't really need, not used anywhere
    """

    name: str
    executable: str
    stdin_file: Optional[str] = None
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    start_file: Optional[str] = None
    target_file: Optional[str] = None
    error_file: Optional[str] = None
    max_running_minutes: Optional[int] = None
    min_arg: Optional[int] = None
    max_arg: Optional[int] = None
    arglist: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    arg_types: list[SchemaItemType] = field(default_factory=list)
    environment: dict[str, Union[int, str]] = field(default_factory=dict)
    exec_env: dict[str, Union[int, str]] = field(default_factory=dict)
    default_mapping: dict[str, Union[int, str]] = field(default_factory=dict)
    private_args: SubstitutionList = field(default_factory=SubstitutionList)
    help_text: str = ""

    default_env: ClassVar[dict[str, str]] = {
        "_ERT_ITERATION_NUMBER": "<ITER>",
        "_ERT_REALIZATION_NUMBER": "<IENS>",
        "_ERT_RUNPATH": "<RUNPATH>",
    }

    def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
        """
        Raise errors pertaining to the environment not being
        as the forward model step requires it to be. For example
        a missing FLOW version.
        """

    def validate_pre_realization_run(
        self, fm_step_json: ForwardModelStepJSON
    ) -> ForwardModelStepJSON:
        """
        Used to validate/modify the job JSON to be run by the forward model step runner.
        It will be called for every joblist.json created.
        """
        return fm_step_json

    def __post_init__(self) -> None:
        # We unescape backslash here to keep backwards compatability ie. If
        # the arglist contains a '\n' we interpret it as a newline.
        self.arglist = [
            s.encode("utf-8", "backslashreplace").decode("unicode_escape")
            for s in self.arglist
        ]

        self.environment.update(ForwardModelStep.default_env)

        if self.stdout_file is None:
            self.stdout_file = f"{self.name}.stdout"
        elif self.stdout_file == "null":
            self.stdout_file = None

        if self.stderr_file is None:
            self.stderr_file = f"{self.name}.stderr"
        elif self.stderr_file == "null":
            self.stderr_file = None

        if self.stdin_file == "null":
            self.stdin_file = None


class ForwardModelStepPlugin(ForwardModelStep):
    def __init__(
        self, name: str, command: list[str], **kwargs: Unpack[ForwardModelStepOptions]
    ):
        if not kwargs:
            kwargs = ForwardModelStepOptions()

        executable = command[0]
        arglist = command[1:]

        stdin_file = kwargs.get("stdin_file")
        stdout_file = kwargs.get("stdout_file")
        stderr_file = kwargs.get("stderr_file")
        start_file = kwargs.get("start_file")
        target_file = kwargs.get("target_file")
        error_file = kwargs.get("error_file")
        max_running_minutes = kwargs.get("max_running_minutes")
        environment = kwargs.get("environment", {}) or {}
        exec_env = kwargs.get("exec_env", {}) or {}
        default_mapping = kwargs.get("default_mapping", {}) or {}

        super().__init__(
            name,
            executable=executable,
            arglist=arglist,
            stdin_file=stdin_file,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            start_file=start_file,
            target_file=target_file,
            error_file=error_file,
            max_running_minutes=max_running_minutes,
            min_arg=0,
            max_arg=0,
            required_keywords=[],
            arg_types=[],
            environment=environment,
            exec_env=exec_env,
            default_mapping=default_mapping,
            private_args=SubstitutionList(),
            help_text="",
        )

    @staticmethod
    @abstractmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        """
        Returns the documentation for the plugin forward model
        """
