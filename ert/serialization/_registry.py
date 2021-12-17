from typing import Tuple

from pyrsistent import pmap
from pyrsistent.typing import PMap

from ._serializer import Serializer, _json_serializer, _yaml_serializer

_registry: PMap[str, Serializer] = pmap(
    {
        "application/json": _json_serializer(),
        "application/x-yaml": _yaml_serializer(),
    }
)


def has_serializer(mime: str) -> bool:
    return mime in _registry


def get_serializer(mime: str) -> Serializer:
    if mime not in _registry:
        raise ValueError(f"no serializer for {mime}")
    return _registry[mime]


def registered_types() -> Tuple[str, ...]:
    return tuple(sorted(_registry.keys()))
