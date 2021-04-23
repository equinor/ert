import sys
from typing import Optional, Dict, Any
from pydantic import root_validator, BaseModel, ValidationError

import ert3

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class _ExperimentConfig(BaseModel):
    validate_all = True
    validate_assignment = True
    extra = "forbid"
    allow_mutation = False
    arbitrary_types_allowed = True


class ExperimentConfig(_ExperimentConfig):
    type: Literal["evaluation", "sensitivity"]
    algorithm: Optional[Literal["one-at-a-time"]]

    @root_validator
    def command_defined(cls, experiment: Dict[str, Any]) -> Dict[str, Any]:
        type_ = experiment.get("type")
        algorithm = experiment.get("algorithm")

        if type_ == "evaluation":
            if algorithm != None:
                raise ValueError("Did not expect algorithm for evaluation experiment")
        elif type_ == "sensitivity":
            if algorithm == None:
                raise ValueError("Expected an algorithm for sensitivity experiments")
        else:
            raise ValueError(f"Unexpected experiment type: {type_}")

        return experiment


def load_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig(**config_dict)
    except ValidationError as err:
        raise ert3.exceptions.ConfigValidationError(str(err), source="experiment")
