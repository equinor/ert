import sys
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type, no_type_check

from pydantic import BaseModel, ValidationError, create_model, root_validator, validator

import ert
import ert.ensemble_evaluator

from ._config_plugin_registry import ConfigPluginRegistry, create_plugged_model

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class SourceNS(str, Enum):
    stochastic = "stochastic"
    storage = "storage"
    resources = "resources"


class _EnsembleConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True


class ForwardModel(_EnsembleConfig):
    stage: str
    driver: Literal["local", "pbs"] = "local"


def _validate_transformation(
    discriminator: str,
    default_for_category: Optional[str],
) -> Callable[[Type[BaseModel], Dict[str, Any]], Dict[str, Any]]:
    def _validator(cls: Type[BaseModel], values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            source = values["source"]
            record = values["record"]
        except KeyError:
            # defer validation of source/record to other, more specific validators
            return values
        try:
            namespace, location = source.split(".", maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"{source} missing at least one dot (.) to form a namespace"
            ) from error

        # no further validation for non-resources. If there is a transformation but it
        # is not defined, remove it, else it would become a default one, which does not
        # make sense for non-resources.
        if namespace != SourceNS.resources:
            if "transformation" in values and not values["transformation"]:
                del values["transformation"]
            return values

        try:
            config = values["transformation"]
        except KeyError as error:
            # if no transformation in config, inject it
            if default_for_category:
                config = values["transformation"] = {
                    discriminator: default_for_category,
                    "location": location,
                }
            else:
                raise ValueError(f"no 'transformation' for input '{record}'") from error

        if "location" in config and config["location"] != location:
            raise ValueError(
                f"two different locations were defined: '{config['location']}'"
                + f" in the 'transformation' config, and in the source '{location}'. "
                + "Either define only the source, or let them be equal."
            )
        config["location"] = location
        return values

    return _validator


class EnsembleInput(_EnsembleConfig):
    """:class:`EnsembleInput` defines inputs over the ensemble. Use it in conjunction
    with :class:`ert3.config.SourceNS` to load resources or sample parameters.

    :class:`EnsembleInput` is the basis for :class:`ert3.config.EnsembleInput` which
    is the plugged-in version of it. Using this model directly will raise a
    :class:`RuntimeError`. Instantiate instead this model using either
    :py:func:`ert3.config.load_ensemble_config` or
    :py:func:`ert3.config.create_ensemble_config`.
    """

    _namespace: SourceNS
    _location: str
    record: str
    source: str

    @no_type_check
    def __init__(self, **data: Any) -> None:
        # isinstance is too lenient for this case, needs to specifically check for
        # type EnsembleInput, specifically not subclasses. So C0123 must be disabled.
        # If this instance is an EnsembleInput, it means no plugins which is no deal.
        if type(self) is EnsembleInput:  # pylint: disable=C0123
            raise RuntimeError(
                "this configuration must be obtained from 'create_ensemble_config'."
            )
        super().__init__(**data)
        parts = data["source"].split(".", maxsplit=1)
        self._namespace = SourceNS(parts[0])
        self._location = parts[1]

    @property
    def source_namespace(self) -> SourceNS:
        return self._namespace

    @property
    def source_location(self) -> str:
        return self._location

    @property
    def direction(self) -> ert.data.TransformationDirection:
        # We know, since this is _input_, that transformations are always going in the
        # TO_RECORD direction.
        return ert.data.TransformationDirection.TO_RECORD


class EnsembleOutput(_EnsembleConfig):
    record: str


class EnsembleConfig(_EnsembleConfig):
    forward_model: ForwardModel
    input: Tuple[EnsembleInput, ...]
    output: Tuple[EnsembleOutput, ...]
    size: Optional[int] = None
    storage_type: str = "ert_storage"
    active_range: Optional[str] = None
    """Specifies list of ranges of realizations that are active.
    Default (``None``) means all realizations are active.
    Empty string means no realizations are active."""

    @validator("active_range")
    def is_active_range_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        ert.ensemble_evaluator.ActiveRange.validate_rangestring(value)
        return value

    @root_validator
    def active_range_vs_size(cls, values):  # type: ignore
        if values.get("active_range") is not None:
            if values.get("size") is None:
                return values
            # If size is not provided, we accept any active_range
            ert.ensemble_evaluator.ActiveRange.validate_rangestring_vs_length(
                values["active_range"], values["size"]
            )
        return values


def create_ensemble_config(
    plugin_registry: ConfigPluginRegistry,
) -> Type[EnsembleConfig]:
    """Return a :class:`EnsembleConfig` with plugged-in configurations."""

    discriminator = plugin_registry.get_descriminator("transformation")
    default_discriminator = plugin_registry.get_default_for_category("transformation")

    input_config = create_plugged_model(
        model_name="PluggedEnsembleInput",
        categories=["transformation"],
        plugin_registry=plugin_registry,
        model_base=EnsembleInput,
        model_module=__name__,
        validators={
            "validate_transformation": root_validator(pre=True, allow_reuse=True)(
                _validate_transformation(
                    discriminator=discriminator,
                    default_for_category=default_discriminator,
                )
            )
        },
    )

    ensemble_config = create_model(
        "PluggedEnsembleConfig",
        __base__=EnsembleConfig,
        __module__=__name__,
        input=(Tuple[input_config, ...], ...),
    )

    return ensemble_config


def load_ensemble_config(
    config_dict: Dict[str, Any], plugin_registry: ConfigPluginRegistry
) -> EnsembleConfig:
    try:
        ensemble_config = create_ensemble_config(plugin_registry=plugin_registry)
        return ensemble_config.parse_obj(config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="ensemble")
