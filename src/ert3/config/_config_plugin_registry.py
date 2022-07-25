import importlib
from collections import defaultdict
from functools import partialmethod
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    NamedTuple,
    Optional,
    Type,
    Union,
    List,
    TYPE_CHECKING,
)
from typing_extensions import Literal


from pydantic import (
    BaseModel,
    create_model,
    Field,
)


class _RegisteredConfig(NamedTuple):
    original_config: Type[BaseModel]
    config: Type[BaseModel]
    factory: Callable[[Type[BaseModel], Type[BaseModel]], Any]
    default_for_category: bool = False


class ConfigPluginRegistry:
    """:class:`ConfigPluginRegistry` is a registry for registering configurations that
    are to be plugged into ert3 configurations."""

    def __init__(self) -> None:
        self._descriminator: Dict[str, str] = {}
        self._registry: Dict[str, Dict[str, _RegisteredConfig]] = {}
        self._base_configs: Dict[str, Type[BaseModel]] = {}

        # Ellipsis (...) is pydantic's way of declaring a required field.
        # Setting this to None, will make it optional. Setting it to
        # Optional[...] will make it required, but allow a None value.
        # https://pydantic-docs.helpmanual.io/usage/models/#required-optional-fields
        # If a category is optional, this will be set to None.
        self._default: DefaultDict[str, Any] = defaultdict(lambda: Ellipsis)

    def register_category(
        self,
        category: str,
        base_config: Type[BaseModel],
        descriminator: str = "type",
        optional: bool = True,
    ) -> None:
        """
        Args:
            category: the category to register. Can only be registered once.
            base_config: A base config for the category. It provides common options
                for all configurations, and all registered configs are expected to
                inherit from this class.
            descriminator: If multiple configurations belong to one category, a
                discriminator must be used such that the configuration system knows
                exactly what configuration to create. See `tagged unions`_.
            optional: whether or not providing configuration in this category is
                optional

        .. _tagged unions: https://pydantic-docs.helpmanual.io/usage/types/
        """
        if category in self._registry:
            raise ValueError(f"Category '{category}' is already registered")
        self._descriminator[category] = descriminator
        self._base_configs[category] = base_config
        self._registry[category] = {}

        if optional:
            self._default[category] = None

    # pylint: disable=too-many-arguments
    def register(
        self,
        name: str,
        category: str,
        config: Type[BaseModel],
        factory: Callable[[Type[BaseModel], Type[BaseModel]], Any],
        is_default_for_category: bool = False,
    ) -> None:
        """Register a configuration for a category.

        Args:
            name: the name of the configuration
            category: a previously registered category
            config: the configuration associated with name
            factory: a factory method that produces something given the full
                configuration. The signature of this method is
                ``f(config, parent_config) -> Any``.
            is_default_for_category: whether or not this name will be the default
                config if no config discrimination could be made.
        """
        if category not in self._registry:
            raise ValueError(
                f"Unknown category '{category}' when registering plugin config '{name}'"
            )
        if name in self._registry[category]:
            raise ValueError(f"{name} is already registered")

        field_definitions: Any = {
            self._descriminator[category]: (Literal[name], Ellipsis)
        }
        config_name = f"Full{config.__name__}"
        full_config = create_model(
            config_name, __base__=config, __module__=__name__, **field_definitions
        )

        # make importable
        mod = importlib.import_module(__name__)
        setattr(mod, config_name, full_config)

        self._registry[category][name] = _RegisteredConfig(
            config=full_config,
            factory=factory,
            default_for_category=is_default_for_category,
            original_config=config,
        )

    def get_factory(
        self, category: str, name: str
    ) -> Callable[[Type[BaseModel], Type[BaseModel]], Any]:
        """Return the factory for this category/name combination."""
        return self._registry[category][name].factory

    def get_descriminator(self, category: str) -> str:
        """Return the discriminator for this category."""
        return self._descriminator[category]

    def get_base_config(self, category: str) -> Type[BaseModel]:
        """Return the base config class for this category."""
        return self._base_configs[category]

    def get_type(self, category: str) -> Any:
        """Return the type for this category. If multiple associated names exist, a
        ``Union`` will be created."""
        if category not in self._registry:
            raise ValueError(f"Unknown category '{category}'")
        values = tuple(o.config for o in self._registry[category].values())
        if not values:
            raise ValueError(
                "Using a plugin field requires at least one registered type, "
                + f"category '{category}' has no registered plugins"
            )

        if len(values) > 1:
            return Union[values]
        else:
            return values[0]

    def get_field(self, category: str) -> Any:
        """Return the field for this category. If multiple associated names exist, a
        descriminator will exist in this field."""
        if category not in self._registry:
            raise ValueError(f"Unknown category '{category}'")
        if not self._registry[category]:
            raise ValueError(
                "Using a plugin field requires at least one registered type, "
                + f"category '{category}' has no registered plugins"
            )

        if len(self._registry[category]) == 1:
            return Field(self._default[category])
        else:
            return Field(
                self._default[category],
                discriminator=self._descriminator[category],
            )

    def get_default_for_category(self, category: str) -> Optional[str]:
        """Return the default for a category when no discrimination could be made."""
        default: Optional[str] = None
        for name, config in self._registry[category].items():
            if config.default_for_category:
                if default:
                    raise RuntimeError(
                        f"two defaults for '{category}': {default} and {name}"
                    )
                default = name
        return default

    def get_original_configs(self, category: str) -> Dict[str, Type[BaseModel]]:
        """Return the field for this category. If multiple associated names exist, a
        descriminator will exist in this field."""
        if category not in self._registry:
            raise ValueError(f"Unknown category '{category}'")
        if not self._registry[category]:
            return {}

        return {
            name: rc.original_config for name, rc in self._registry[category].items()
        }


def _getter_template(
    self: Any, category: str, optional: bool, plugin_registry: ConfigPluginRegistry
) -> Any:
    """Return an instance of the factory method associated with this category, given
    that this method is bound to a configuration in this category."""
    config_instance = getattr(self, category)
    if optional and not config_instance:
        return None
    elif not optional and not config_instance:
        raise ValueError("no config, but was required for '{category}' configuration")
    descriminator_value = getattr(
        config_instance, plugin_registry.get_descriminator(category=category)
    )
    return plugin_registry.get_factory(category=category, name=descriminator_value)(
        config_instance, self
    )


# https://mypy.readthedocs.io/en/latest/runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    Classmethod = classmethod[Any]  # pylint: disable=unsubscriptable-object
else:
    Classmethod = classmethod


# pylint: disable=too-many-arguments
def create_plugged_model(
    model_name: str,
    categories: List[str],
    plugin_registry: ConfigPluginRegistry,
    model_base: Optional[Type[BaseModel]] = None,
    model_module: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    validators: Optional[Dict[str, Classmethod]] = None,
    docs: bool = False,
) -> Type[BaseModel]:
    """Create a plugged model ``model_name`` with the given ``categories``.

    Args:
        model_name: name of this model
        categories: categories to plug-in
        plugin_registry: plugin registry where categories are defined
        model_base: base class of this pydantic model
        model_module: specify to what module this model will be associated
        extra_fields: Fields are defined by either a tuple of the form
            ``(<type>, <default value>)`` or just a default value. See
            https://pydantic-docs.helpmanual.io/usage/models/#dynamic-model-creation
        validators: a mapping of validators, see
            https://pydantic-docs.helpmanual.io/usage/models/#dynamic-model-creation
            and ``__validators__``
    """
    fields: Dict[str, Any] = extra_fields if extra_fields else {}
    model_attrs: Dict[str, Any] = {}

    for category in categories:
        category_field = plugin_registry.get_field(category)
        category_type = plugin_registry.get_type(category)
        fields[category] = (Any, Any) if docs else (category_type, category_field)
        model_attrs[f"get_{category}_instance"] = partialmethod(
            _getter_template,
            category=category,
            optional=category_field.default != Ellipsis,
            plugin_registry=plugin_registry,
        )

    model = create_model(
        model_name,
        __base__=model_base,
        __module__=model_module if model_module else __name__,
        __validators__=validators if validators else {},
        **fields,
    )

    for attr, value in model_attrs.items():
        setattr(model, attr, value)
    return model
