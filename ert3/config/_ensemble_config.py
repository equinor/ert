import sys
from enum import Enum
from typing import Any, Dict, Optional, Tuple, no_type_check

from pydantic import BaseModel, ValidationError, validator

import ert
from ._validator import ensure_mime

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


class Input(_EnsembleConfig):
    _namespace: SourceNS
    _location: str
    source: str
    record: str
    mime: str = ""
    is_directory: Optional[bool]

    # This is copied from the pydantic documentation, but is apparently not
    # popular with mypy, so ignore entire block.
    @no_type_check  # type: ignore
    def __init__(self, **data: Any) -> None:
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

    @validator("source")
    def split_source(cls, v: str) -> str:
        parts = v.split(".", maxsplit=1)
        if not len(parts) == 2:
            raise ValueError(f"{v} missing at least one dot (.) to form a namespace")
        return v

    _ensure_mime = validator("mime", allow_reuse=True)(ensure_mime("source"))


class Output(_EnsembleConfig):
    record: str


class EnsembleConfig(_EnsembleConfig):
    forward_model: ForwardModel
    input: Tuple[Input, ...]
    output: Tuple[Output, ...]
    size: Optional[int] = None
    storage_type: str = "ert_storage"


def load_ensemble_config(config_dict: Dict[str, Any]) -> EnsembleConfig:
    try:
        return EnsembleConfig(**config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="ensemble")
