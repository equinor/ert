import sys
from typing import Optional, Dict, Any
from pydantic import root_validator, BaseModel, ValidationError
from pydantic.class_validators import validator

import ert

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class _ExperimentConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


class ExperimentConfig(_ExperimentConfig):
    type: Literal["evaluation", "sensitivity"]
    algorithm: Optional[Literal["one-at-a-time"]]
    tail: Optional[float]

    @root_validator
    def command_defined(cls, experiment: Dict[str, Any]) -> Dict[str, Any]:
        type_ = experiment.get("type")
        algorithm = experiment.get("algorithm")
        tail = experiment.get("tail")

        if type_ == "evaluation":
            if algorithm != None:
                raise ValueError("Did not expect algorithm for evaluation experiment")
            if tail != None:
                raise ValueError("Did not expect tail for evaluation experiment")
        elif type_ == "sensitivity":
            if algorithm == None:
                raise ValueError("Expected an algorithm for sensitivity experiments")
        else:
            raise ValueError(f"Unexpected experiment type: {type_}")

        return experiment

    @validator("tail")
    def _ensure_valid_tail(cls, tail: Optional[float]) -> Optional[float]:
        if tail is not None:
            if tail <= 0:
                raise ValueError("Tail cannot be <= 0")
            if tail >= 1:
                raise ValueError("Tail cannot be >= 1")
            return tail
        return tail


def load_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig(**config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="experiment")
