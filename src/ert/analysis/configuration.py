from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, Union

from pydantic import (
    BaseModel,
    conlist,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ert.analysis.update import Parameter


class Observation(BaseModel):
    name: str
    index_list: List[int] = []


if TYPE_CHECKING:
    ConstrainedObservationList = List[Observation]
else:
    ConstrainedObservationList = conlist(Observation, min_length=1)


class UpdateStep(BaseModel):
    name: str
    observations: ConstrainedObservationList
    parameters: List[Parameter] = []

    @field_validator("parameters", mode="before")
    def transform_parameters(cls, parameters: Any) -> List[Parameter]:
        if parameters is None:
            return []
        if not isinstance(parameters, (list, tuple)):
            raise ValueError("value is not a valid list")
        values = []
        for parameter in parameters:
            if isinstance(parameter, str):
                values.append(Parameter(parameter))
            elif len(parameter) == 1:
                values.append(Parameter(parameter[0]))
            elif len(parameter) == 2:
                values.append(Parameter(parameter[0], parameter[1]))
        return values

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """
        Because the user only has to specify parameters or row_scaling_parameters, we
        check that they have at least configured one parameter
        """
        if not self.parameters:
            raise ValueError("Must provide at least one parameter")
        return self

    @field_validator("observations", mode="before")
    def check_arguments(
        cls,
        observations: Union[
            str,
            Dict[str, Union[str, List[int]]],
            List[Union[str, List[int]]],
            Tuple[Union[str, List[int]]],
        ],
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """Because most of the time the users will configure observations as only a name
        we convert positional arguments to named arguments"""
        values: List[Dict[str, Union[str, List[int]]]] = []
        for observation in observations:
            if isinstance(observation, str):
                values.append({"name": observation})
            elif isinstance(observation, dict):
                values.append(observation)
            else:
                assert isinstance(observation, (tuple, list))
                if len(observation) == 1:
                    name = observation[0]
                    assert isinstance(name, str)
                    values.append({"name": name})
                elif len(observation) == 2:
                    name, index_list = observation
                    assert isinstance(name, str)
                    assert isinstance(index_list, list)
                    values.append({"name": name, "index_list": index_list})
                else:
                    raise ValueError(
                        f"Unexpected observation length {len(observation)}"
                    )
        return values


if TYPE_CHECKING:
    ConstrainedUpdateStepList = List[UpdateStep]
else:
    ConstrainedUpdateStepList = conlist(UpdateStep, min_length=1)


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
            for parameter in update_step.parameters:
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
