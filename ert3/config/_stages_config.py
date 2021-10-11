from importlib.abc import Loader
import importlib.util
from typing import Callable, Tuple, cast, Union, Dict, Any, Mapping
from types import MappingProxyType
from collections import OrderedDict
from pydantic import BaseModel, FilePath, ValidationError, validator
import ert
from ._validator import ensure_mime


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


class Record(_StagesConfig):
    record: str
    location: str
    mime: str = ""
    is_directory: bool = False

    _ensure_record_mime = validator("mime", allow_reuse=True)(ensure_mime("location"))


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath
    mime: str = ""

    _ensure_transportablecommand_mime = validator("mime", allow_reuse=True)(
        ensure_mime("location")
    )


# mypy ignore missing parameter for generic type
class IndexedOrderedDict(OrderedDict):  # type: ignore
    """Extend OrderedDict to add support for accessing elements by their index."""

    def __getitem__(self, attr: Union[str, int]) -> Any:
        if isinstance(attr, str):
            return super().__getitem__(attr)
        return self[list(self.keys())[attr]]


def _create_record_mapping(records: Tuple[Dict[str, str], ...]) -> Mapping[str, Record]:
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
        _create_record_mapping
    )
    _set_output = validator("output", pre=True, always=True, allow_reuse=True)(
        _create_record_mapping
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


Step = Union[Function, Unix]
