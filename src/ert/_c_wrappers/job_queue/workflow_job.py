import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from ert._c_wrappers.config import ConfigParser, ConfigValidationError, ContentTypeEnum
from ert._c_wrappers.job_queue import ErtScript, ExternalErtScript, FunctionErtScript
from ert._clib.job_kw import type_from_kw

from .ert_plugin import ErtPlugin

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain

    ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]


def _workflow_job_config_parser() -> ConfigParser:
    parser = ConfigParser()
    parser.add("MIN_ARG", value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(1, 1)
    parser.add("MAX_ARG", value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(1, 1)
    parser.add(
        "EXECUTABLE", value_type=ContentTypeEnum.CONFIG_EXECUTABLE
    ).set_argc_minmax(1, 1)
    parser.add("SCRIPT", value_type=ContentTypeEnum.CONFIG_PATH).set_argc_minmax(1, 1)
    parser.add("FUNCTION").set_argc_minmax(1, 1)
    parser.add("INTERNAL", value_type=ContentTypeEnum.CONFIG_BOOL).set_argc_minmax(1, 1)
    item = parser.add("ARG_TYPE")
    item.set_argc_minmax(2, 2)
    item.iset_type(0, ContentTypeEnum.CONFIG_INT)
    item.initSelection(1, ["STRING", "INT", "FLOAT", "BOOL"])
    return parser


_config_parser = _workflow_job_config_parser()


class ErtScriptLoadFailure(ValueError):
    pass


@dataclass
class WorkflowJob:
    name: str
    internal: bool
    min_args: Optional[int]
    max_args: Optional[int]
    arg_types: List[ContentTypeEnum]
    executable: Optional[str]
    script: Optional[str]
    function: Optional[str]

    def __post_init__(self):
        self.__running = False
        self._ert_script: Optional[type] = None
        if self.script is not None and self.internal:
            try:
                self._ert_script = ErtScript.loadScriptFromFile(
                    self.script,
                )  # type: ignore
            # Bare Exception here as we have no control
            # of exceptions in the loaded ErtScript
            except Exception as err:  # noqa
                raise ErtScriptLoadFailure(
                    f"Failed to load {self.name}: {err}"
                ) from err

    @staticmethod
    def _make_arg_types_list(
        config_content, max_arg: Optional[int]
    ) -> List[ContentTypeEnum]:
        arg_types_dict = defaultdict(lambda: ContentTypeEnum.CONFIG_STRING)
        if max_arg is not None:
            arg_types_dict[max_arg - 1] = ContentTypeEnum.CONFIG_STRING
        for arg in config_content["ARG_TYPE"]:
            arg_types_dict[arg[0]] = ContentTypeEnum(type_from_kw(arg[1]))
        if arg_types_dict:
            return [
                arg_types_dict[j]
                for j in range(max(i for i in arg_types_dict.keys()) + 1)
            ]
        else:
            return []

    @classmethod
    def fromFile(cls, config_file, name=None):
        if os.path.isfile(config_file) and os.access(config_file, os.R_OK):
            if name is None:
                name = os.path.basename(config_file)

            content = _config_parser.parse(config_file)

            def optional_get(key):
                return content.getValue(key) if content.hasKey(key) else None

            max_arg = optional_get("MAX_ARG")

            return cls(
                name,
                optional_get("INTERNAL"),
                optional_get("MIN_ARG"),
                max_arg,
                cls._make_arg_types_list(content, max_arg),
                optional_get("EXECUTABLE"),
                optional_get("SCRIPT"),
                optional_get("FUNCTION"),
            )
        else:
            raise ConfigValidationError(f"Could not open config_file:{config_file!r}")

    def isPlugin(self) -> bool:
        if self._ert_script is not None:
            return issubclass(self._ert_script, ErtPlugin)
        return False

    def argumentTypes(self) -> List["ContentTypes"]:
        def content_to_type(c: Optional[ContentTypeEnum]):
            if c == ContentTypeEnum.CONFIG_BOOL:
                return bool
            if c == ContentTypeEnum.CONFIG_FLOAT:
                return float
            if c == ContentTypeEnum.CONFIG_INT:
                return int
            if c == ContentTypeEnum.CONFIG_STRING:
                return str
            raise ValueError(f"Unknown job type {c} in {self}")

        return list(map(content_to_type, self.arg_types))

    @property
    def execution_type(self):
        if self.internal and self.script is not None:
            return "internal python"
        elif self.internal:
            return "internal C"
        return "external"

    def run(self, ert: "EnKFMain", arguments: List[Any]) -> Any:
        self.__running = True
        if self.min_args and len(arguments) < self.min_args:
            raise ValueError(
                f"The job: {self.name} requires at least "
                f"{self.min_args} arguments, {len(arguments)} given."
            )

        if self.max_args and self.max_args < len(arguments):
            raise ValueError(
                f"The job: {self.name} can only have "
                f"{self.max_args} arguments, {len(arguments)} given."
            )

        if self._ert_script is not None:
            self.__script = self._ert_script(ert)
        elif self.internal and self.function is not None:
            self.__script = FunctionErtScript(
                ert,
                self.function,
                self.argumentTypes(),
                argument_count=len(arguments),
            )
        elif not self.internal:
            self.__script = ExternalErtScript(ert, self.executable)
        else:
            raise UserWarning("Unknown script type!")
        result = self.__script.initializeAndRun(self.argumentTypes(), arguments)
        self.__running = False
        return result

    def cancel(self) -> None:
        if self.__script is not None:
            self.__script.cancel()

    def isRunning(self) -> bool:
        return self.__running

    def isCancelled(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling isCancelled")
        return self.__script.isCancelled()

    def hasFailed(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling hasFailed")
        return self.__script.hasFailed()

    def stdoutdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stdoutdata")
        return self.__script.stdoutdata

    def stderrdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stderrdata")
        return self.__script.stderrdata
