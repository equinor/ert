from pathlib import Path
from typing import List, Optional, Type

from pydantic import BaseModel, validator

from ert3.config import ConfigPluginRegistry
from ert3.config.plugins import TransformationConfigBase
from ert3.config._validator import ensure_mime

from ert3.plugins.plugin_manager import hook_implementation

from ert.data import CopyTransformation
from ert.data import SerializationTransformation
from ert.data import TarTransformation
from ert.data import EclSumTransformation


def transformation_factory(
    config: Type[BaseModel], parent_config: Type[BaseModel]
) -> TransformationConfigBase:
    if isinstance(config, CopyTransformationConfig):
        return CopyTransformation(
            location=Path(config.location),
            direction=parent_config.direction,
        )
    elif isinstance(config, SerializationTransformationConfig):
        return SerializationTransformation(
            location=Path(config.location),
            mime=config.mime,
            direction=parent_config.direction,
        )
    elif isinstance(config, SummaryTransformationConfig):
        if not config.smry_keys:
            raise ValueError(f"no smry_keys on '{config}'")
        return EclSumTransformation(
            location=Path(config.location),
            smry_keys=config.smry_keys,
            direction=parent_config.direction,
        )
    elif isinstance(config, DirectoryTransformationConfig):
        return TarTransformation(
            location=Path(config.location),
            direction=parent_config.direction,
        )
    else:
        raise ValueError(
            f"Unknown config type {type(config)} for config instance {config}"
        )


class CopyTransformationConfig(TransformationConfigBase):
    pass


class SerializationTransformationConfig(TransformationConfigBase):
    mime: str = ""

    _ensure_mime = validator("mime", allow_reuse=True)(ensure_mime("location"))


class SummaryTransformationConfig(TransformationConfigBase):
    smry_keys: Optional[List[str]] = None


class DirectoryTransformationConfig(TransformationConfigBase):
    pass


@hook_implementation  # type: ignore  # pluggy lacks type information
def configs(registry: ConfigPluginRegistry) -> None:
    registry.register(
        name="copy",
        category="transformation",
        config=CopyTransformationConfig,
        factory=transformation_factory,
        is_default_for_category=True,
    )
    registry.register(
        name="serialization",
        category="transformation",
        config=SerializationTransformationConfig,
        factory=transformation_factory,
    )
    registry.register(
        name="summary",
        category="transformation",
        config=SummaryTransformationConfig,
        factory=transformation_factory,
    )
    registry.register(
        name="directory",
        category="transformation",
        config=DirectoryTransformationConfig,
        factory=transformation_factory,
    )
