from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from pydantic import BaseModel, conlist, root_validator, validator
from typing_extensions import Self

from .update import Parameter, RowScalingParameter


class Observation(BaseModel):
    name: str
    index_list: List[int] = []


if TYPE_CHECKING:
    ConstrainedObservationList = List[Observation]
else:
    ConstrainedObservationList = conlist(Observation, min_items=1)


class UpdateStep(BaseModel):
    name: str
    observations: ConstrainedObservationList
    parameters: List[Parameter] = []
    row_scaling_parameters: List[RowScalingParameter] = []

    @root_validator(pre=True)
    def transform_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parameters = values.get("parameters", [])
        if not isinstance(parameters, (list, tuple)):
            raise ValueError("value is not a valid list")
        values["parameters"] = [
            (p,) if isinstance(p, str) else tuple(p) for p in parameters
        ]
        row_scaling_parameters = values.get("row_scaling_parameters", [])
        if not isinstance(row_scaling_parameters, (list, tuple)):
            raise ValueError("value is not a valid list")
        values["row_scaling_parameters"] = [
            (p,) if isinstance(p, str) else tuple(p) for p in row_scaling_parameters
        ]
        return values

    @root_validator(skip_on_failure=True)
    def check_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
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
    def check_arguments(cls, observation: Union[str, List[str], Dict[str, Any]]) -> Any:
        """Because most of the time the users will configure observations as only a name
        we convert positional arguments to named arguments"""
        if isinstance(observation, str):
            return {"name": observation}
        elif isinstance(observation, dict):
            return observation
        else:
            if len(observation) == 1:
                return {"name": observation[0]}
            elif len(observation) == 2:
                name, index_list = observation
                return {"name": name, "index_list": index_list}
            else:
                raise ValueError(f"Unexpected observation length {len(observation)}")

    def observation_config(self) -> List[Tuple[str, Optional[List[int]]]]:
        return [
            (observation.name, observation.index_list)
            for observation in self.observations
        ]


if TYPE_CHECKING:
    ConstrainedUpdateStepList = List[UpdateStep]
else:
    ConstrainedUpdateStepList = conlist(UpdateStep, min_items=1)


class UpdateConfiguration(BaseModel):
    update_steps: ConstrainedUpdateStepList

    def __iter__(self) -> Iterator[UpdateStep]:  # type: ignore
        yield from self.update_steps

    def __getitem__(self, item: int) -> UpdateStep:
        return self.update_steps[item]

    def __len__(self) -> int:
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

    @classmethod
    def global_update_step(cls, observations: List[str], parameters: List[str]) -> Self:
        global_update_step = [
            UpdateStep(
                name="ALL_ACTIVE",
                observations=observations,
                parameters=parameters,
            )
        ]
        return cls(update_steps=global_update_step)
