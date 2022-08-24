# pylint: disable=import-error,no-member,not-callable
# (false positives)
from typing import Dict, List

from pydantic import BaseModel, conlist, root_validator, validator

from ert._c_wrappers.enkf.row_scaling import RowScaling
from ert._clib.update import Parameter, RowScalingParameter


@classmethod  # type: ignore
def __get_validators__(cls):
    yield cls.validate


@classmethod  # type: ignore
def validate_parameter(cls, v):
    if isinstance(v, str):
        return cls(v)
    else:
        return cls(*v)


@classmethod  # type: ignore
def validate_row_scaling_parameter(cls, v):
    if not isinstance(v[1], RowScaling):
        raise TypeError(f"Expected RowScaling type, got: {type(v[1])}")
    return cls(*v)


Parameter.__get_validators__ = __get_validators__
Parameter.validate = validate_parameter

RowScalingParameter.__get_validators__ = __get_validators__
RowScalingParameter.validate = validate_row_scaling_parameter


__all__ = ["Parameter", "RowScalingParameter"]


class Observation(BaseModel):
    name: str
    index_list: List[int] = []


class UpdateStep(BaseModel):
    name: str
    observations: conlist(Observation, min_items=1)  # type: ignore
    parameters: List[Parameter] = []
    row_scaling_parameters: List[RowScalingParameter] = []

    @root_validator(skip_on_failure=True)
    def check_parameters(cls, values):
        """
        Because the user only has to specify parameters or row_scaling_parameters, we
        check that they have at least configured one parameter
        """
        parameters = values.get("parameters", [])
        row_scaling_parameters = values.get("row_scaling_parameters", [])
        if len(parameters) + len(row_scaling_parameters) == 0:
            raise ValueError("Must provide at least one parameter")
        return values

    @validator("observations", each_item=True, pre=True)
    def check_arguments(cls, observation):
        """Because most of the time the users will configure observations as only a name
        we convert positional arguments to named arguments"""
        if isinstance(observation, str):
            return {"name": observation}
        elif not isinstance(observation, Dict):
            if len(observation) == 1:
                return {"name": observation[0]}
            elif len(observation) == 2:
                name, index_list = observation
                return {"name": name, "index_list": index_list}
        return observation

    def observation_config(self):
        return [
            (observation.name, observation.index_list)
            for observation in self.observations
        ]


class UpdateConfiguration(BaseModel):
    update_steps: conlist(UpdateStep, min_items=1)  # type: ignore

    def __iter__(self):
        yield from self.update_steps

    def __getitem__(self, item):
        return self.update_steps[item]

    def __len__(self):
        return len(self.update_steps)

    def context_validate(
        self, valid_observations: List[str], valid_parameters: List[str]
    ) -> None:
        errors = []
        for update_step in self.update_steps:
            for observation in update_step.observations:
                if observation.name not in valid_observations:
                    errors.append(
                        f"Observation: {observation} not in valid observations"
                    )
            for parameter in (
                update_step.parameters + update_step.row_scaling_parameters
            ):
                if parameter.name not in valid_parameters:
                    errors.append(f"Parameter: {parameter} not in valid parameters")
        if errors:
            raise ValueError(
                f"Update configuration not valid, "
                f"valid observations: {valid_observations}, "
                f"valid parameters: {valid_parameters}, errors: {errors}"
            )
