from importlib.abc import Loader
import importlib.util
import mimetypes
import sys
from typing import Callable, List, cast, Union

from pydantic import BaseModel, FilePath, ValidationError, root_validator, validator
import ert3

_DEFAULT_RECORD_MIME_TYPE: str = "application/json"
_DEFAULT_CMD_MIME_TYPE: str = "application/octet-stream"


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


class _Step(_StagesConfig):
    name: str
    input: List[InputRecord]
    output: List[OutputRecord]

    # backwards compatible
    @root_validator(pre=True)
    def set_step(cls, values):
        if "type" in values:
            del values["type"]
        return values


class Function(_Step):
    function: Callable

    @validator("function", pre=True)
    def function_is_callable(cls, value) -> Callable:
        return _import_from(value)


class Unix(_Step):
    script: List[str]
    transportable_commands: List[TransportableCommand]


class StagesConfig(BaseModel):
    __root__: List[Union[Function, Unix]]

    def step_from_key(self, key: str) -> Union[Function, Unix, None]:
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
