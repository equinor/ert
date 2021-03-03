from importlib.abc import Loader
import importlib.util
import mimetypes
import sys
from typing import Callable, List, Optional, cast

from pydantic import BaseModel, FilePath, ValidationError, root_validator, validator
import ert3

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_DEFAULT_RECORD_MIME_TYPE = "application/json"
_DEFAULT_CMD_MIME_TYPE = "application/octet-stream"


def _import_from(path) -> Callable:
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
        func = getattr(module, func)
    except AttributeError:
        raise ImportError(name=func, path=module_str)
    return func


def _ensure_mime(cls, field, values):
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


class _StagesConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


class InputRecord(_StagesConfig):
    record: str
    location: str
    mime: str = ""

    _ensure_mime = validator("mime", allow_reuse=True)(_ensure_mime)


class OutputRecord(_StagesConfig):
    record: str
    location: str
    mime: str = ""

    _ensure_mime = validator("mime", allow_reuse=True)(_ensure_mime)


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath
    mime: str = ""

    _ensure_mime = validator("mime", allow_reuse=True)(_ensure_mime)


class Step(_StagesConfig):
    name: str
    type: Literal["unix", "function"]
    script: Optional[List[str]] = []
    input: List[InputRecord]
    output: List[OutputRecord]
    transportable_commands: Optional[List[TransportableCommand]] = []
    function: Optional[Callable]

    @root_validator
    def check_defined(cls, step):
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
    def function_is_valid(cls, function: str, values) -> Optional[Callable]:
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

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)


def load_stages_config(config_dict):
    try:
        return StagesConfig.parse_obj(config_dict)
    except ValidationError as err:
        raise ert3.exceptions.ConfigValidationError(str(err), source="stages")
