from typing import List
from typing_extensions import Literal
from pydantic.dataclasses import dataclass
from pydantic import BaseModel


class _EnsembleConfig(BaseModel):
    validate_all = True
    validate_assignment = True
    extra = "forbid"
    allow_mutation = False
    arbitrary_types_allowed = True


class ForwardModel(_EnsembleConfig):
    stages: List[str]
    driver: Literal["local", "pbs"] = "local"


class Input(_EnsembleConfig):
    source: str
    record: str


class EnsembleConfig(_EnsembleConfig):
    forward_model: ForwardModel
    input: List[Input]
    size: int


def load_ensemble_config(config_dict):
    return EnsembleConfig(**config_dict)
