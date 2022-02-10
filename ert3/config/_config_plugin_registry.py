from collections import namedtuple
from typing import Any, Callable, Dict, Type, Union
from typing_extensions import Literal

from pydantic import BaseModel, Field, create_model


_RegisteredConfig = namedtuple("_RegisteredConfig", "config factory")


class ConfigPluginRegistry:
    def __init__(self) -> None:
        self._descriminator: Dict[str, str] = {}
        self._registry: Dict[str, Dict[str, _RegisteredConfig]] = {}

    def register_category(self, category: str, descriminator: str = "type"):
        if category in self._registry:
            raise ValueError(f"Category '{category}' is already registered")
        self._descriminator[category] = descriminator
        self._registry[category] = {}

    def register(
        self,
        name: str,
        category: str,
        config: Type[BaseModel],
        factory: Callable[[Type[BaseModel]], Any],
    ):
        if not category in self._registry:
            raise ValueError(
                f"Unknown category '{category}' when registering plugin config '{name}'"
            )
        if name in self._registry[category]:
            raise ValueError(f"{name} is already registered")

        field_definitions = {self._descriminator[category]: (Literal[name], ...)}
        full_config = create_model(
            f"Full{config.__name__}", __base__=config, **field_definitions
        )

        self._registry[category][name] = _RegisteredConfig(
            config=full_config, factory=factory
        )

    def get_factory(self, category: str, name: str):
        return self._registry[category][name].factory

    def get_descriminator(self, category: str):
        return self._descriminator[category]

    def get_type(self, category: str):
        if not category in self._registry:
            raise ValueError(f"Unknown category '{category}'")
        values = tuple(o.config for o in self._registry[category].values())
        if not values:
            raise ValueError(
                f"Using a plugin field requires at least one registered type, category '{category}' has no registered plugins"
            )

        if len(values) > 1:
            return Union[values]  # type: ignore
        else:
            return values[0]

    def get_field(self, category: str):
        if not category in self._registry:
            raise ValueError(f"Unknown category '{category}'")
        if not self._registry[category]:
            raise ValueError(
                f"Using a plugin field requires at least one registered type, category '{category}' has no registered plugins"
            )

        if len(self._registry[category]) == 1:
            return Field(...)
        else:
            return Field(..., discriminator=self._descriminator[category])
