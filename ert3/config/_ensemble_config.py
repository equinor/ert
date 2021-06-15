import sys
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ValidationError

from ert3.exceptions import ConfigValidationError

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class _EnsembleConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


class ForwardModel(_EnsembleConfig):
    stage: str
    driver: Literal["local", "pbs"] = "local"


class Input(_EnsembleConfig):
    source: str
    record: str


class EnsembleConfig(_EnsembleConfig):
    forward_model: ForwardModel
    inputs: Tuple[Input, ...]
    size: Optional[int] = None


def load_ensemble_config(config_dict: Dict[str, Any]) -> EnsembleConfig:
    try:
        return EnsembleConfig(**config_dict)
    except ValidationError as err:
        raise ConfigValidationError(str(err), source="ensemble")
