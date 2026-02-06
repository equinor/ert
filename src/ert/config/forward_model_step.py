from __future__ import annotations

import inspect
import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    cast,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import TypedDict, Unpack

from ..base_model_context import BaseModelWithContextSupport
from .parsing import ConfigValidationError, ConfigWarning, SchemaItemType

if TYPE_CHECKING:
    from ert.plugins import ErtRuntimePlugins

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
        max_running_minutes: Maximum runtime in minutes. If the forward model step
            takes longer than this, the step is requested to be cancelled.
    """

    name: str
    executable: str
    target_file: str | None
    error_file: str | None
    start_file: str | None
    stdout: str | None
    stderr: str | None
    stdin: str | None
    argList: list[str]
    environment: dict[str, str] | None
    max_running_minutes: int | None


class ForwardModelStepOptions(TypedDict, total=False):
    stdin_file: NotRequired[str]
    stdout_file: NotRequired[str]
    stderr_file: NotRequired[str]
    start_file: NotRequired[str]
    target_file: NotRequired[str]
    error_file: NotRequired[str]
    max_running_minutes: NotRequired[int]
    environment: NotRequired[dict[str, str]]
    default_mapping: NotRequired[dict[str, str]]
    required_keywords: NotRequired[list[str]]


def _get_source_package() -> str:
    """Return the top-level package name of the calling forward model step.

    Finds the documentation() call (stack[2]) under the forward model step class
    and return its parent module
    """
    stack = inspect.stack()
    if len(stack) > 2:
        caller_frame = stack[2]
        caller_module = inspect.getmodule(caller_frame.frame)
        if caller_module:
            return caller_module.__name__.split(".")[0]
    return "not found"


class ForwardModelStepDocumentation(BaseModel):
    config_file: str | None = Field(default=None)
    source_package: str = Field(default_factory=_get_source_package)
    source_function_name: str = Field(default="ert")
    description: str = Field(default="No description")
    examples: str = Field(default="No examples")
    category: Annotated[
        str,
        Field(
            default="Uncategorized",
            examples=[
                "utility.file_system",
                "simulators.reservoir",
                "modelling.reservoir",
                "utility.templating",
            ],
        ),
    ]


class ForwardModelStep(BaseModelWithContextSupport):
    """
    Holds information to execute one step of a forward model

    Attributes:
        executable: The name of the executable to be run
        stdin_file: File where this forward model step's stdin is written
        stdout_file: File where this forward model step's stdout is written
        stderr_file: File where this forward model step's stderr is written
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
            takes longer than this, the step is requested to be cancelled.
        min_arg: The minimum number of arguments
        max_arg: The maximum number of arguments
        arglist: The arglist with which the executable is invoked
        required_keywords: Keywords that are required to be supplied by the user
        arg_types: Types of user-provided arguments, this list must have indices
            corresponding to the number of args that are user specified
            and thus also subject to substitution.
        environment: Dictionary representing environment variables to inject into the
            environment of the forward model step run
        default_mapping: Default values for optional arguments provided by the user.
            For example { "A": "default_A" }
        private_args: A dictionary of user-provided keyword arguments.
            For example, if the user provides <A>=2, the dictionary will contain
            { "A": "2" }
    """

    name: str
    executable: str
    stdin_file: str | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None
    start_file: str | None = None
    target_file: str | None = None
    error_file: str | None = None
    max_running_minutes: int | None = None
    min_arg: int | None = None
    max_arg: int | None = None
    arglist: list[str] = Field(default_factory=list)
    required_keywords: list[str] = Field(default_factory=list)
    arg_types: list[SchemaItemType] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    default_mapping: dict[str, str] = Field(default_factory=dict)
    private_args: dict[str, str] = Field(default_factory=dict)

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
        Used to validate/modify the step JSON to be run by the forward model step
        runner. It will be called for every joblist.json created.
        """
        return fm_step_json

    def check_required_keywords(self) -> None:
        """
        Raises ConfigValidationError if not all required keywords are in
        private_args
        """
        missing_keywords = set(self.required_keywords).difference(self.private_args)
        if missing_keywords:
            plural = "s" if len(missing_keywords) > 1 else ""
            raise ConfigValidationError.with_context(
                f"Required keyword{plural} {', '.join(sorted(missing_keywords))} "
                f"not found for forward model step {self.name}",
                self.name,
            )

    @model_validator(mode="after")
    def set_default_env(self) -> Self:
        self.environment.update(ForwardModelStep.default_env)
        return self

    @model_validator(mode="after")
    def set_stdout_stderr_defaults(self) -> Self:
        for attr in ["stdout_file", "stderr_file"]:
            value = getattr(self, attr, None)
            if value is None:
                setattr(self, attr, f"{self.name}.{attr.split('_')[0]}")
            elif value == "null":
                setattr(self, attr, None)
        return self

    @field_validator("stdin_file", mode="after")
    @classmethod
    def set_stdin_file(cls, v: str | None) -> str | None:
        return None if v == "null" else v


class UserInstalledForwardModelStep(ForwardModelStep):
    """
    Represents a forward model step installed by a user via the ERT Config
    forward model step format provided via the INSTALL_JOB keyword.
    User-installed forward model steps serialize with their full configuration,
    unlike site-installed steps which only serialize as references.
    """

    type: Literal["user_installed"] = "user_installed"


class _SerializedSiteInstalledForwardModelStep(TypedDict):
    type: Literal["site_installed"]
    name: str
    private_args: dict[str, str]


class SiteInstalledForwardModelStep(ForwardModelStep):
    """
    Represents a forward model step installed via external plugins.
    Instances of this class serialize only as references to the plugin by name, and
    the user-provided private_args, allowing them to dynamically update when plugins
    change, rather than being locked to a specific executable at serialization time.
    """

    type: Literal["site_installed"] = "site_installed"

    @model_serializer(mode="plain")
    def serialize_model(self) -> _SerializedSiteInstalledForwardModelStep:
        return {
            "type": "site_installed",
            "name": self.name,
            "private_args": self.private_args,
        }

    @model_validator(mode="before")
    @classmethod
    def deserialize_model(
        cls, values: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        runtime_plugins = cast("ErtRuntimePlugins", info.context)
        name = values["name"]

        if runtime_plugins is None:
            if values.get("type") == "site_installed":
                msg = (
                    f"Trying to find site-installed forward model step {name} "
                    f"without site plugins. This forward model must be loaded "
                    f"with ERT site plugins available."
                )
                raise KeyError(msg)
            return values

        if name not in runtime_plugins.installed_forward_model_steps:
            msg = (
                f"Expected forward model step {name} to be installed "
                f"via plugins, but it was not found. Please check that "
                f"your python environment has it installed."
            )
            raise KeyError(msg)
        site_installed_fm = runtime_plugins.installed_forward_model_steps[name]

        # Intent: copy the site installed forward model to this instance.
        # bypassing the model_serializer
        site_fm_instance = {
            k: getattr(site_installed_fm, k)
            for k in SiteInstalledForwardModelStep.model_fields
        }

        return site_fm_instance | (
            {"private_args": values["private_args"]} if "private_args" in values else {}
        )


SiteOrUserForwardModelStep = Annotated[
    (UserInstalledForwardModelStep | SiteInstalledForwardModelStep),
    Field(discriminator="type"),
]


class ForwardModelStepPlugin(SiteInstalledForwardModelStep):
    def __init__(
        self, name: str, command: list[str], **kwargs: Unpack[ForwardModelStepOptions]
    ) -> None:
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
        default_mapping = kwargs.get("default_mapping", {}) or {}
        required_keywords = kwargs.get("required_keywords", []) or []

        super().__init__(
            name=name,
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
            required_keywords=required_keywords,
            arg_types=[],
            environment=environment,
            default_mapping=default_mapping,
            private_args={},
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        """
        Returns the documentation for the plugin forward model
        """
        return None


class ForwardModelJSON(TypedDict):
    global_environment: dict[str, str]
    config_path: str
    config_file: str
    jobList: list[ForwardModelStepJSON]
    run_id: str | None
    ert_pid: str
    max_runtime: int | None
