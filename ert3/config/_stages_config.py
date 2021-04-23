from importlib.abc import Loader
import importlib.util
import mimetypes
import sys
from typing import Callable, List, Optional, cast, Dict, Any

from pydantic import BaseModel, FilePath, ValidationError, root_validator, validator
import ert3

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_DEFAULT_RECORD_MIME_TYPE = "application/json"
_DEFAULT_CMD_MIME_TYPE = "application/octet-stream"


def _import_from(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError("Function should be defined as module:function")
    module_str, func = path.split(":")
    spec = importlib.util.find_spec(module_str)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_str}'")
    module = importlib.util.module_from_spec(spec)
    # A loader should always have been set, and it is assumed it a PEP 302
    # compliant Loader. A cast is made to make this clear to the typing system.
    cast(Loader, spec.loader).exec_module(module)
    try:
        return cast(Callable[..., Any], getattr(module, func))
    except AttributeError:
        raise ImportError(name=func, path=module_str)


class _StagesConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


def _ensure_mime(cls: _StagesConfig, field: str, values: Dict[str, Any]) -> str:
    if field:
        return field
    guess = mimetypes.guess_type(str(values.get("location", "")))[0]
    if guess:
        return guess
    return (
        _DEFAULT_CMD_MIME_TYPE
        if cls == TransportableCommand
        else _DEFAULT_RECORD_MIME_TYPE
    )


class InputRecord(_StagesConfig):
    record: str
    location: str
    mime: str = ""

    @validator("mime")
    def _ensure_input_record_mime(cls, field: str, values: Dict[str, Any]) -> str:
        return _ensure_mime(cls, field, values)


class OutputRecord(_StagesConfig):
    record: str
    location: str
    mime: str = ""

    @validator("mime")
    def _ensure_output_record_mime(cls, field: str, values: Dict[str, Any]) -> str:
        return _ensure_mime(cls, field, values)


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath
    mime: str = ""

    @validator("mime")
    def _ensure_transportable_command_mime(
        cls, field: str, values: Dict[str, Any]
    ) -> str:
        return _ensure_mime(cls, field, values)


class Step(_StagesConfig):
    name: str
    type: Literal["unix", "function"]
    script: Optional[List[str]] = []
    input: List[InputRecord]
    output: List[OutputRecord]
    transportable_commands: Optional[List[TransportableCommand]] = []
    function: Optional[Callable[..., Any]]

    @root_validator
    def check_defined(cls, step: Dict[str, Any]) -> Dict[str, Any]:
        cmd_names = [cmd.name for cmd in step.get("transportable_commands", [])]
        script_lines = step.get("script", [])
        if step.get("type") == "function":
            if cmd_names:
                raise ValueError("Commands defined for a function stage")
            if script_lines:
                raise ValueError("Scripts defined for a function stage")
            if not step.get("function"):
                raise ValueError("No function defined")
        return step

    @validator("function", pre=True)
    def function_is_valid(
        cls, function: str, values: Dict[str, Any]
    ) -> Optional[Callable[..., Any]]:
        step_type = values.get("type")
        if step_type != "function" and function:
            raise ValueError(f"Function defined for {step_type} step")
        if function is None:
            return None
        return _import_from(function)


class StagesConfig(BaseModel):
    __root__: List[Step]

    def step_from_key(self, key: str) -> Optional[Step]:
        return next((step for step in self if step.name == key), None)

    def __iter__(self):  # type: ignore
        return iter(self.__root__)

    def __getitem__(self, item):  # type: ignore
        return self.__root__[item]

    def __len__(self):  # type: ignore
        return len(self.__root__)


def load_stages_config(config_dict: Dict[str, Any]) -> StagesConfig:
    try:
        return StagesConfig.parse_obj(config_dict)
    except ValidationError as err:
        raise ert3.exceptions.ConfigValidationError(str(err), source="stages")
