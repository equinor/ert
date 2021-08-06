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
    algorithm: Optional[Literal["one-at-a-time", "fast"]]
    tail: Optional[float]
    harmonics: Optional[int]
    sample_size: Optional[int]

    @root_validator
    def command_defined(cls, experiment: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=R0912
        type_ = experiment.get("type")
        algorithm = experiment.get("algorithm")
        tail = experiment.get("tail")
        harmonics = experiment.get("harmonics")
        sample_size = experiment.get("sample_size")

        if type_ == "evaluation":
            if algorithm is not None:
                raise ValueError("Did not expect algorithm for evaluation experiment")
            if tail is not None:
                raise ValueError("Did not expect tail for evaluation experiment")
            if harmonics is not None:
                raise ValueError("Did not expect harmonics for evaluation experiment")
            if sample_size is not None:
                raise ValueError("Did not expect sample_size for evaluation experiment")
        elif type_ == "sensitivity":
            if algorithm == None:
                raise ValueError("Expected an algorithm for sensitivity experiments")
            if algorithm == "one-at-a-time":
                if harmonics is not None:
                    raise ValueError(
                        "Did not expect harmonics for one-at-a-time algorithm"
                    )
                if sample_size is not None:
                    raise ValueError(
                        "Did not expect sample_size for one-at-a-time algorithm"
                    )
            if algorithm == "fast":
                if tail is not None:
                    raise ValueError("Did not expect tail for fast algorithm")
                if harmonics is None:
                    raise ValueError("Expected harmonics for fast algorithm")
                if sample_size is None:
                    raise ValueError("Expected sample_size for fast algorithm")
                if sample_size < 4 * pow(harmonics, 2) + 1:
                    raise ValueError("Sample_size must be >= 4 * harmonics^2 + 1")
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

    @validator("harmonics")
    def _ensure_valid_harmonics(cls, harmonics: Optional[float]) -> Optional[float]:
        if harmonics is not None:
            if harmonics <= 0:
                raise ValueError("Harmonics cannot be <= 0")
            return harmonics
        return harmonics

    @validator("sample_size")
    def _ensure_valid_sample_size(cls, sample_size: Optional[float]) -> Optional[float]:
        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError("Sample_size cannot be <= 0")
            return sample_size
        return sample_size


def load_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig(**config_dict)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="experiment")
