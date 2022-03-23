import importlib.util
from collections import OrderedDict
from importlib.abc import Loader
import pathlib
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Tuple,
    Union,
    cast,
    Type,
    Optional,
)

from pydantic import (
    BaseModel,
    FilePath,
    ValidationError,
    root_validator,
    validator,
)


import ert
from ._config_plugin_registry import ConfigPluginRegistry, create_plugged_model

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
    """the file path of the script or executable on disk. During execution, the script
    or executable can be located in the execution environment's ``bin`` directory with
    the name equivalent to the ``Pathlib.name`` property, i.e. the final component
    excluding root and parent folders.

    If ``location`` is set to ``my/bins/script.py``, in the execution environment its
    path will be ``bin/script.py``.
    """
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


def _default_transformation_discriminator(
    category: str,
    discriminator: str,
    default_for_category: Optional[str],
) -> Callable[[Type[BaseModel], Dict[str, Any]], Dict[str, Any]]:
    """Create a validator such that default_for_category for category is injected into
    the configuration if it was not provided to the discriminator field and if
    default_for_category is not empty."""

    def _validator(cls: Type[BaseModel], values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            config = values[category]
        except KeyError:
            return values
        if discriminator not in config and default_for_category:
            config[discriminator] = default_for_category
        return values

    return _validator


class StageIO(_StagesConfig):
    name: str
    direction: ert.data.TransformationDirection

    @validator("direction", pre=True)
    def _valid_direction(cls, v: str) -> ert.data.TransformationDirection:
        return ert.data.TransformationDirection.from_direction(v)


def create_stages_config(plugin_registry: ConfigPluginRegistry) -> "Type[StagesConfig]":
    """Return a :class:`StagesConfig` with plugged-in configurations."""
    discriminator = plugin_registry.get_descriminator("transformation")
    default_discriminator = plugin_registry.get_default_for_category("transformation")

    stage_io = create_plugged_model(
        model_name="PluggedStageIO",
        categories=["transformation"],
        plugin_registry=plugin_registry,
        model_base=StageIO,
        model_module=__name__,
        validators={
            "discriminator_default": root_validator(pre=True, allow_reuse=True)(
                _default_transformation_discriminator(
                    category="transformation",
                    discriminator=discriminator,
                    default_for_category=default_discriminator,
                )
            ),
        },
    )

    # duck punching _Step to bridge static and dynamic config definitions. StageIO
    # exists only at run-time, but _Step (and subclasses) should be static.
    _Step._stageio_cls = stage_io

    # Returning the StagesConfig class to underline that it needs some dynamic mutation.
    return StagesConfig


class _Step(_StagesConfig):

    # See create_stages_config
    _stageio_cls: Any
    name: str
    input: MappingProxyType  # type: ignore
    output: MappingProxyType  # type: ignore

    @classmethod
    def _create_io_mapping(
        cls,
        ios: List[Dict[str, str]],
        direction: str,
    ) -> Mapping[str, Type[_StagesConfig]]:
        if not hasattr(cls, "_stageio_cls"):
            raise RuntimeError(
                "Step configuration must be obtained through 'create_stages_config'."
            )

        for io in ios:
            if "direction" not in io:
                io["direction"] = direction

        ordered_dict = IndexedOrderedDict(
            {io["name"]: cls._stageio_cls(**io) for io in ios}
        )

        proxy = MappingProxyType(ordered_dict)
        return proxy

    @validator("input", pre=True, always=True, allow_reuse=True)
    def _create_input_mapping(
        cls, ios: List[Dict[str, str]]
    ) -> Mapping[str, Type[_StagesConfig]]:
        return cls._create_io_mapping(ios, direction="from_record")

    @validator("output", pre=True, always=True, allow_reuse=True)
    def _create_output_mapping(
        cls, ios: List[Dict[str, str]]
    ) -> Mapping[str, Type[_StagesConfig]]:
        return cls._create_io_mapping(ios, direction="to_record")


class Function(_Step):
    function: Callable  # type: ignore

    @validator("function", pre=True)
    def function_is_callable(cls, value) -> Callable:  # type: ignore
        return _import_from(value)


class Unix(_Step):
    script: Tuple[str, ...]
    transportable_commands: Tuple[TransportableCommand, ...]

    def command_location(self, name: str) -> pathlib.Path:
        """Return a path to the location of the command with the given name."""
        return next(
            (cmd.location for cmd in self.transportable_commands if cmd.name == name),
            pathlib.Path(name),
        )

    def command_final_path_component(self, name: str) -> pathlib.Path:
        """Return a path using the ``pathlib.Path.name`` property of the command
        location for the command with the given name."""
        return pathlib.Path(self.command_location(name).name)

    @root_validator
    def _ensure_ios_has_transformation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for io in ("input", "output"):
            if io not in values:
                continue
            for name, io_ in values[io].items():
                if not io_.transformation:
                    raise ValueError(f"io '{name}' had no transformation")
        return values


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


def load_stages_config(
    config_dict: Dict[str, Any], plugin_registry: ConfigPluginRegistry
) -> StagesConfig:
    stages_config = create_stages_config(plugin_registry=plugin_registry)
    try:
        return stages_config.parse_obj(config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="stages")
