from importlib.abc import Loader
import importlib.util
import mimetypes
from typing import Callable, Tuple, Type, cast, Union, Dict, Any, Mapping
from types import MappingProxyType
from collections import OrderedDict
from pydantic import BaseModel, FilePath, ValidationError, validator
import ert


DEFAULT_RECORD_MIME_TYPE: str = "application/octet-stream"
DEFAULT_CMD_MIME_TYPE: str = "application/octet-stream"


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


def _ensure_mime(cls: Type[_StagesConfig], field: str, values: Dict[str, Any]) -> str:
    if field:
        return field
    guess = mimetypes.guess_type(str(values.get("location", "")))[0]
    if guess:
        if not ert.serialization.has_serializer(guess):
            return DEFAULT_RECORD_MIME_TYPE
        return guess
    return (
        DEFAULT_CMD_MIME_TYPE
        if cls == TransportableCommand
        else DEFAULT_RECORD_MIME_TYPE
    )


class Record(_StagesConfig):
    record: str
    location: str
    mime: str = ""

    _ensure_record_mime = validator("mime", allow_reuse=True)(_ensure_mime)


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath
    mime: str = ""

    _ensure_transportablecommand_mime = validator("mime", allow_reuse=True)(
        _ensure_mime
    )


# mypy ignore missing parameter for generic type
class IndexedOrderedDict(OrderedDict):  # type: ignore
    """Ordered dict with custom getitem-method"""

    def __getitem__(self, attr: Any) -> Any:
        if isinstance(attr, str):
            return super().__getitem__(attr)
        return self[list(self.keys())[attr]]


def _set_dict_from_list(
    cls: Type[_StagesConfig], records: Tuple[Dict[str, str], ...]
) -> Mapping[str, Record]:
    """Grab record-names in records and use as keys"""
    ordered_dict = IndexedOrderedDict(
        {record["record"]: Record(**record) for record in records}
    )
    proxy = MappingProxyType(ordered_dict)
    return proxy


class _Step(_StagesConfig):
    name: str
    input: MappingProxyType  # type: ignore
    output: MappingProxyType  # type: ignore

    _set_input = validator("input", pre=True, always=True, allow_reuse=True)(
        _set_dict_from_list
    )
    _set_output = validator("output", pre=True, always=True, allow_reuse=True)(
        _set_dict_from_list
    )


class Function(_Step):
    function: Callable  # type: ignore

    @validator("function", pre=True)
    def function_is_callable(cls, value) -> Callable:  # type: ignore
        return _import_from(value)


class Unix(_Step):
    script: Tuple[str, ...]
    transportable_commands: Tuple[TransportableCommand, ...]


class StagesConfig(BaseModel):
    __root__: Tuple[Union[Function, Unix], ...]

    def step_from_key(self, key: str) -> Union[Function, Unix, None]:
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
        raise ert.exceptions.ConfigValidationError(str(err), source="stages")
