from typing import List, Optional

try:
    # Will only work from Python 3.8
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic.dataclasses import dataclass
from pydantic import root_validator, BaseModel


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
    def command_defined(cls, experiment):
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


def load_experiment_config(config_dict):
    return ExperimentConfig(**config_dict)
