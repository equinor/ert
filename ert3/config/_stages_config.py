import importlib.util
from collections import OrderedDict
from functools import partialmethod
from importlib.abc import Loader
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

from pydantic import BaseModel, FilePath, ValidationError, create_model, validator

import ert
from ert3.config import ConfigPluginRegistry

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


def create_stage_io(plugin_registry) -> type:
    def getter_template(self, category):
        config_instance = getattr(self, category)
        descriminator_value = getattr(
            config_instance, plugin_registry.get_descriminator(category=category)
        )
        return plugin_registry.get_factory(category=category, name=descriminator_value)(
            config_instance
        )

    stage_io_fields = {"name": (str, None)}
    stage_io_methods = {}
    for category in ["transformation"]:
        stage_io_fields[category] = (
            plugin_registry.get_type(category),
            plugin_registry.get_field(category),
        )
        stage_io_methods[f"get_{category}_instance"] = partialmethod(
            getter_template, category=category
        )

    stage_io = create_model(
        "StageIO",
        __base__=_StagesConfig,
        **stage_io_fields,
    )
    for name, method in stage_io_methods.items():
        setattr(stage_io, name, method)

    return stage_io


def get_configs(plugin_registry):
    """
    We now need the create configs dynamically,
    as we would like to control the schema created based on the plugins we provide.
    This will allow us to specify a subset of plugins we want to have effect at runtime,
    such as only using configs from ert in tests.
    """

    StageIO = create_stage_io(plugin_registry=plugin_registry)

    def _create_io_mapping(ios: Tuple[Dict[str, str], ...]) -> Mapping[str, StageIO]:
        ordered_dict = IndexedOrderedDict({io["record"]: StageIO(**io) for io in ios})
        proxy = MappingProxyType(ordered_dict)
        return proxy

    class _Step(_StagesConfig):
        name: str
        input: MappingProxyType  # type: ignore
        output: MappingProxyType  # type: ignore

        _set_input = validator("input", pre=True, always=True, allow_reuse=True)(
            _create_io_mapping
        )
        _set_output = validator("output", pre=True, always=True, allow_reuse=True)(
            _create_io_mapping
        )

    class Function(_Step):
        function: Callable  # type: ignore

        @validator("function", pre=True)
        def function_is_callable(cls, value) -> Callable:  # type: ignore
            return _import_from(value)

    class Unix(_Step):
        script: Tuple[str, ...]
        transportable_commands: Tuple[TransportableCommand, ...]

    Step = Union[Function, Unix]

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

    return StagesConfig, Step


def register_plugins() -> ConfigPluginRegistry:
    """
    This would normally be controlled by a module that calls the pluggy hooks,
    similar to ert_shared/plugins/plugin_manager.py.
    """
    plugin_registry = ConfigPluginRegistry()
    plugin_registry.register_category(category="transformation")
    plugin_registry.register(
        name="file",
        category="transformation",
        config=FileTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    plugin_registry.register(
        name="directory",
        category="transformation",
        config=DirectoryTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    plugin_registry.register(
        name="summary",
        category="transformation",
        config=SummaryTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    return plugin_registry


def load_stages_config(
    config_dict: Dict[str, Any], plugin_registry=ConfigPluginRegistry
):
    StagesConfig, _ = get_configs(plugin_registry=plugin_registry)
    try:
        return StagesConfig.parse_obj(config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="stages")
